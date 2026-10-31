"""Export Mistral's Timestral forecaster to ONNX for onnx-gomlx, with a PyTorch reference output.

usage:
    uv run python export.py --model mistralai/Timestral-28M --out model-28m
    uv run python export.py --model mistralai/Timestral-123M --out model-123m

Writes to --out:
    timestral.onnx  weights inline; input `context` (batch, C) float32, NaN = missing;
                    output `quantiles` (batch, H, 21); dynamic batch
    meta.json       model, context_steps, horizon_steps, quantile_levels
    context.f32     C little-endian float32: a synthetic 5-min series (daily cycle, noise, a gap)
    reference.f32   (2, H, 21) little-endian float32: PyTorch output for the batch [context, -context]

The export needs two patches (torch.export can't handle them as is):
  1. torch.Tensor.cummax -> log-step running max: ONNX has no CumMax (only used on int64 indices, so exact).
  2. timestral.forecaster.ffill_with_mask without its data-dependent early exit `if mask.all(): return inputs`.
The exported graph is TimestralForecaster._forecast_raw without its data-dependent NaN check; symmetric
averaging (forecast of -context, mirrored) is done by the caller with the second row of the batch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from timestral import TimestralForecaster

STEPS_PER_DAY = 288  # 5-min buckets
STEPS_PER_HOUR = 12


class _CumMax(tuple):
    @property
    def values(self):
        return self[0]


def _running_max(self: torch.Tensor, dim: int = -1):
    assert dim in (-1, self.dim() - 1)
    x = self.int() if self.dtype == torch.bool else self
    out, k, n = x, 1, x.shape[-1]
    while k < n:  # log-step doubling: ceil(log2 n) elementwise maxima
        shifted = torch.cat([out[..., :1].expand(*out.shape[:-1], k), out[..., :-k]], dim=-1)
        out = torch.maximum(out, shifted)
        k *= 2
    return _CumMax((out.bool() if self.dtype == torch.bool else out, None))


def _ffill_with_mask(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(inputs.shape[-1], device=inputs.device).expand_as(inputs)
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    idx = _running_max(idx).values
    return torch.nan_to_num(torch.gather(inputs, dim=-1, index=idx), nan=0.0)


def patch_for_export() -> None:
    import timestral.forecaster as tf

    torch.Tensor.cummax = _running_max
    tf.ffill_with_mask = _ffill_with_mask


class RawForecast(torch.nn.Module):
    """(B, C) context, NaN for missing -> (B, H, Q) quantiles: one pass, no symmetric averaging."""

    def __init__(self, f: TimestralForecaster, horizon: int) -> None:
        super().__init__()
        self.f = f
        self.model = f.model
        self.horizon = horizon

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # TimestralForecaster._as_context + _forecast_raw without the NaN check (not exportable)
        f = self.f
        x = context[:, None, :]
        obs = torch.isfinite(x)
        x = torch.where(obs, x, torch.zeros_like(x))
        rows, mask, role_ids, kf, kf_mask, n_tgt = f._assemble_rows(x, obs, self.horizon, None, None)
        q = f._block_ar_decode(rows, mask, self.horizon, role_ids, kf, kf_mask)[:, :n_tgt]
        if f.quantile_real_cap_k > 0:
            q = f._cap_quantiles(q, rows[:, :n_tgt], mask[:, :n_tgt])
        return q[:, 0]


def synthetic_context(steps: int) -> np.ndarray:
    """Temperature-like 5-min series: daily cycle + slow drift + noise, with a gap of missing values."""
    rng = np.random.default_rng(0)
    t = np.arange(steps)
    x = 20 + 5 * np.sin(2 * np.pi * t / STEPS_PER_DAY) + 0.5 * np.sin(2 * np.pi * t / (7 * STEPS_PER_DAY))
    x += rng.normal(0, 0.2, steps)
    x[-500:-300] = np.nan
    return x.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mistralai/Timestral-28M")
    p.add_argument("--context-days", type=int, default=42)
    p.add_argument("--horizon-hours", type=int, default=72)
    p.add_argument("--out", type=Path, default=Path("model-28m"))
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--tolerance", type=float, default=1e-3, help="max abs diff ONNX Runtime vs PyTorch")
    args = p.parse_args()

    C = args.context_days * STEPS_PER_DAY
    H = args.horizon_hours * STEPS_PER_HOUR
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "timestral.onnx"

    f = TimestralForecaster.from_pretrained(args.model)
    m = RawForecast(f, H).eval()
    ctx = synthetic_context(C)
    batch = np.stack([ctx, -ctx])
    with torch.inference_mode():
        t0 = time.time()
        ref = m(torch.from_numpy(batch)).numpy()
        print(f"pytorch (unpatched) batch 2: {time.time() - t0:.2f}s")

    patch_for_export()
    with torch.no_grad():
        torch.onnx.export(
            m,
            (torch.from_numpy(batch),),
            str(onnx_path),
            input_names=["context"],
            output_names=["quantiles"],
            dynamic_shapes={"context": {0: torch.export.Dim("batch", min=1, max=64)}},
            opset_version=args.opset,
            dynamo=True,
            external_data=False,
        )
    print(f"exported {onnx_path}: {onnx_path.stat().st_size / 1e6:.0f} MB")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.time()
    out = sess.run(None, {"context": batch})[0]
    diff = float(np.abs(out - ref).max())
    print(f"onnxruntime batch 2: {time.time() - t0:.2f}s, max abs diff vs PyTorch {diff:.2e}")
    if not diff <= args.tolerance:
        raise SystemExit(f"FAILED: max abs diff {diff} > {args.tolerance}")

    ctx.astype("<f4").tofile(args.out / "context.f32")
    ref.astype("<f4").tofile(args.out / "reference.f32")
    meta = {
        "model": args.model,
        "context_steps": C,
        "horizon_steps": H,
        "quantile_levels": list(f.quantile_levels),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()