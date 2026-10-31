Experimental: benchmarks using github.com/gomlx/onnx-gomlx to convert Timestral models

# Timestral model benchmark on Go backend

- Discussion: https://github.com/gomlx/onnx-gomlx/pull/94 
- Original benchmark original description: https://gist.github.com/arigon/9a99eb74a81b7f9f4532b8f0a6ed98b0
- Timestral GitHub page: https://github.com/mistralai/timestral

## Benchmarks

Using `go1.27.1`

### 2026-10-31 Parallel `Concatenate`, `ReduceSum`, unary ops & in-place `FusedSoftmax`/`FusedActivation` (Step 6)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | best    | 1.577s | 39.545s | 25.1 | 2.7e-05 | 1.98 GB |
| amd9950x3d/AVX512 | model-123m/ | best    | 0.212s | 3.478s | 16.4 | 2.3e-05 | 1.90 GB |
| amd9950x3d/nosimd | model-28m/ | best    | 0.405s | 9.198s | 22.7 | 3.4e-05 | 0.75 GB |
| amd9950x3d/AVX512 | model-28m/ | best    | 0.068s | 0.705s | 10.4 | 3.8e-05 | 0.75 GB |

### 2026-10-31 2D-collapsible broadcast & in-place `Where` optimization (Step 5)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | best    | 1.601s | 39.678s | 24.8 | 5.1e-05 | 2.03 GB |
| amd9950x3d/AVX512 | model-123m/ | best    | 0.220s | 3.538s | 16.1 | 2.3e-05 | 1.93 GB |
| amd9950x3d/nosimd | model-28m/ | best    | 0.395s | 9.224s | 23.3 | 3.8e-05 | 0.71 GB |
| amd9950x3d/AVX512 | model-28m/ | best    | 0.065s | 0.578s | 8.9 | 3.8e-05 | 0.71 GB |

### 2026-10-31 3-part `BroadcastMid` & parallel binary/comparison ops (Step 4)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 1.658s | 39.458s | 23.8 | 5.1e-05 | 2.08 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.232s | 3.395s | 14.6 | 2.3e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.425s | 9.306s | 21.9 | 3.8e-05 | 0.82 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.079s | 0.819s | 10.3 | 3.8e-05 | 0.79 GB |

### 2026-10-31 Canonical axis collapsing & parallel `Transpose` fast paths (Step 3)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 1.794s | 39.832s | 22.2 | 5.1e-05 | 2.06 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.322s | 3.182s | 9.9 | 2.3e-05 | 2.08 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.486s | 9.250s | 19.0 | 3.8e-05 | 0.84 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.110s | 0.704s | 6.4 | 3.8e-05 | 0.82 GB |

### 2026-10-31 Contiguous block-copy & parallel `Slice` (Step 2)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 1.926s | 39.643s | 20.6 | 5.1e-05 | 2.13 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.452s | 3.122s | 6.9 | 2.3e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.521s | 9.307s | 17.9 | 3.8e-05 | 0.81 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.156s | 0.815s | 5.2 | 3.8e-05 | 0.80 GB |

### 2026-10-31 Finer-grained MatMul & Scheduler chunking (Causes 2 & 3)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 2.004s | 39.246s | 19.6 | 5.1e-05 | 2.11 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.510s | 3.116s | 6.1 | 2.3e-05 | 2.12 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.545s | 9.420s | 17.3 | 3.8e-05 | 0.86 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.181s | 0.824s | 4.5 | 3.8e-05 | 0.83 GB |

### 2026-10-31 Defaulting to `OpsExecutionDynamic`

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 2.038s | 39.267s | 19.3 | 5.1e-05 | 2.03 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.507s | 2.837s | 5.6 | 1.5e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.594s | 9.268s | 15.6 | 3.8e-05 | 0.88 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.171s | 0.653s | 3.8 | 3.6e-05 | 0.77 GB |

### 2026-10-31 After new Go backend scheduler

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 2.332s | 38.944s | 16.7 | 5.1e-05 | 2.11 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.678s | 2.441s | 3.6 | 1.5e-05 | 2.10 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.707s | 9.332s | 13.2 | 3.8e-05 | 0.83 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.258s | 0.800s | 3.1 | 3.6e-05 | 0.80 GB |

### 2026-10-29

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5   | 1.897s | 37.940s | 20.0 | 5.1e-05 | 2.18 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5   | 0.496s | 3.026s | 6.1 | 2.5e-05 | 2.22 GB |
| amd9950x3d/nosimd | model-28m/ | run 5   | 0.541s | 9.251s | 17.1 | 3.8e-05 | 0.92 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5   | 0.187s | 0.935s | 5.0 | 4.4e-05 | 1.02 GB |
