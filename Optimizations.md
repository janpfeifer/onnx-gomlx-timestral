# Timestral Go Backend Optimization Plan & Profiling Analysis

## 1. Baseline & Regression Overview

Benchmark commands (run from `./benchmarks/timestral/` on AMD Ryzen 9 9950X3D, `GOMAXPROCS=32`):

```bash
GOEXPERIMENT=simd GOMLX_BACKEND=go go run . -obs="amd9950x3d/AVX512" -dir model-28m/
GOEXPERIMENT=simd GOMLX_BACKEND=go go run . -obs="amd9950x3d/AVX512" -dir model-123m/
GOEXPERIMENT=nosimd GOMLX_BACKEND=go go run . -obs="amd9950x3d/nosimd" -dir model-28m/
GOEXPERIMENT=nosimd GOMLX_BACKEND=go go run . -obs="amd9950x3d/nosimd" -dir model-123m/
```

### Comparison Before & After Go Backend Scheduler Modernization (`acb9402` vs `a8d59a4`)

| Date / Version | Obs | Model | Run | Wall | Est. CPU Time (`Wall × Cores`) | Cores | Max RSS |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **2026-10-29** (Before #45/#46) | `amd9950x3d/AVX512` | `model-28m/` | run 5 | **0.187s** | ~0.935s | 5.0 | 1.02 GB |
| **2026-10-31** (After #45/#46) | `amd9950x3d/AVX512` | `model-28m/` | run 5 | **0.258s** (+38%) | ~0.800s | 3.1 | 0.80 GB |
| **2026-10-29** (Before #45/#46) | `amd9950x3d/AVX512` | `model-123m/` | run 5 | **0.496s** | ~3.026s | 6.1 | 2.22 GB |
| **2026-10-31** (After #45/#46) | `amd9950x3d/AVX512` | `model-123m/` | run 5 | **0.678s** (+37%) | ~2.441s | 3.6 | 2.10 GB |
| **2026-10-29** (Before #45/#46) | `amd9950x3d/nosimd` | `model-28m/` | run 5 | **0.541s** | ~9.251s | 17.1 | 0.92 GB |
| **2026-10-31** (After #45/#46) | `amd9950x3d/nosimd` | `model-28m/` | run 5 | **0.707s** (+31%) | ~9.332s | 13.2 | 0.83 GB |
| **2026-10-29** (Before #45/#46) | `amd9950x3d/nosimd` | `model-123m/` | run 5 | **1.897s** | ~37.940s | 20.0 | 2.18 GB |
| **2026-10-31** (After #45/#46) | `amd9950x3d/nosimd` | `model-123m/` | run 5 | **2.332s** (+23%) | ~38.944s | 16.7 | 2.11 GB |

Notice that in `AVX512` (`GOEXPERIMENT=simd`), **total CPU time actually decreased** after the scheduler change (`~0.94s -> ~0.80s` on `28m`, `~3.03s -> ~2.44s` on `123m`), while **Wall time increased by ~37–38%** because average core utilization dropped (`5.0 -> 3.1` cores on `28m`, `6.1 -> 3.6` cores on `123m`). In `nosimd`, total CPU time stayed roughly flat (`~38s -> ~39s`) while core utilization dropped (`20.0 -> 16.7` cores on `123m`, `17.1 -> 13.2` on `28m`).

---

## 2. Per-`OpType` Wall-Clock Breakdown

Measured on a single warm inference pass (`GOMAXPROCS=32`, sequential op execution):

| Operation Category | `28m` SIMD (`242.1 ms`) | `123m` SIMD (`685.3 ms`) | `28m` NoSIMD (`680.0 ms`) | `123m` NoSIMD (`2401.1 ms`) |
| :--- | ---: | ---: | ---: | ---: |
| **`DotGeneral` + `FusedDense`** | 38.1 ms (**15.7%**) | 122.4 ms (**17.9%**) | 403.3 ms (**59.3%**) | 1690.4 ms (**70.4%**) |
| **`Add` + `Mul` + `NotEqual`** | 70.3 ms (**29.1%**) | 203.5 ms (**29.7%**) | 91.9 ms (**13.5%**) | 251.7 ms (**10.5%**) |
| **`Slice`** | 59.3 ms (**24.5%**) | 164.5 ms (**24.0%**) | 68.8 ms (**10.1%**) | 168.6 ms (**7.0%**) |
| **`Transpose`** | 52.3 ms (**21.6%**) | 148.4 ms (**21.7%**) | 57.3 ms (**8.4%**) | 151.8 ms (**6.3%**) |
| **`FusedSoftmax`** | 6.4 ms (**2.6%**) | 17.7 ms (**2.6%**) | 15.2 ms (**2.2%**) | 31.3 ms (**1.3%**) |
| **`Where`** | 3.4 ms (**1.4%**) | 7.6 ms (**1.1%**) | 23.0 ms (**3.4%**) | 63.0 ms (**2.6%**) |
| **`FusedActivation`** (`Gelu`/`Swish`) | 1.8 ms (**0.7%**) | 4.7 ms (**0.7%**) | 7.4 ms (**1.1%**) | 20.4 ms (**0.9%**) |
| **`Concatenate`** | 2.4 ms (**1.0%**) | 6.2 ms (**0.9%**) | 2.7 ms (**0.4%**) | 6.2 ms (**0.3%**) |
| **Other** (`Reshape`, `BroadcastInDim`, `ReduceMax`, `Sqrt`, `Div`, etc.) | 8.1 ms (**3.3%**) | 10.3 ms (**1.5%**) | 10.4 ms (**1.5%**) | 17.7 ms (**0.7%**) |

### Top Individual Op Signatures in `model-123m` (SIMD vs NoSIMD)

| Op & Signature | Calls | `123m` SIMD | `123m` NoSIMD | Notes |
| :--- | ---: | ---: | ---: | :--- |
| `Add((Float32)[2, 10, 378, 378], (Float32)[2, 1, 378, 378])` | 20 | **72.21 ms** | **73.32 ms** | Hits `BroadcastGeneral` (1-core scalar `ZipIterator`!) |
| `Mul((Float32)[2, 405, 10, 64], (Float32)[2, 405, 1, 64])` | 40 | **62.43 ms** | **62.39 ms** | Hits `BroadcastGeneral` (1-core scalar `ZipIterator`!) |
| `Transpose([2, 405, 3, 64, 10], perm=[2, 0, 1, 4, 3])` | 20 | **45.28 ms** | **47.82 ms** | Hits 1-core `TransposeIterator.Next` |
| `Transpose([405, 2, 3, 64, 10], perm=[2, 0, 1, 4, 3])` | 20 | **42.33 ms** | **43.76 ms** | Hits 1-core `TransposeIterator.Next` |
| `Slice((Float32)[2, 405, 3424] -> [2, 405, 1712])` | 40 | **39.77 ms** | **39.59 ms** | 1-core element-by-element loop despite 1712-float contiguous rows |
| `NotEqual((Float32)[2, 10, 378, 378], [2, 10, 378, 378])` | 20 | **30.71 ms** | **30.42 ms** | 1-core scalar loop over 2.86M elements/call |
| `Slice((Float32)[2, 405, 10, 64] -> [2, 405, 10, 32])` | 80 | **29.74 ms** | **30.74 ms** | 1-core element-by-element loop despite 32-float contiguous blocks |
| `DotGeneral((Float32)[2, 405, 640], [640, 3424])` | 20 | **28.56 ms** | **476.62 ms** | `M=810`: `AppendWorkItems` creates 26 items (6 cores idle) |
| `FusedDense((Float32)[2, 405, 640], [640, 1920], [1, 1, 1920])` | 20 | **18.23 ms** | **268.25 ms** | `M=810`: `AppendWorkItems` creates 26 items (6 cores idle) |
| `Slice((Float32)[2, 405, 1920] -> [2, 405, 1024])` | 20 | **14.37 ms** | **13.99 ms** | 1-core element-by-element loop despite 1024-float contiguous rows |
| `Transpose([20, 378, 64], perm=[0, 2, 1])` | 20 | **13.79 ms** | **14.12 ms** | Batched 2D transpose misses `dispatchFastTranspose` |
| `Transpose([20, 405, 64], perm=[0, 2, 1])` | 20 | **13.70 ms** | **14.54 ms** | Batched 2D transpose misses `dispatchFastTranspose` |
| `DotGeneral((Float32)[2, 405, 1712], [1712, 640])` | 20 | **13.38 ms** | **213.66 ms** | `M=810, K=1712, N=640` |
| `DotGeneral((Float32)[2, 10, 378, 378], [2, 10, 378, 64])` | 20 | **12.50 ms** | **140.47 ms** | `batchSize=20`: `20×2=40` items -> Wave 2 uses only 8/32 cores |
| `Transpose([4050, 2, 64], perm=[0, 2, 1])` | 40 | **11.14 ms** | **11.16 ms** | Batched 2D transpose misses `dispatchFastTranspose` |
| `Mul((Float32)[2, 405, 1712], (Float32)[2, 405, 1712])` | 20 | **9.87 ms** | **12.01 ms** | 1-core SIMD / 1-core scalar (1.39M elements/call) |
| `Where((Bool)[2, 10, 378, 378], (Float32), [2, 10, 378, 378])` | 20 | **4.68 ms** | **52.53 ms** | Reuses `onFalse` in-place, yet writes `out[i] = onFalse[i]` |

### Key Takeaway for `GOEXPERIMENT=simd`

In `GOEXPERIMENT=simd`, `DotGeneral + FusedDense` is already very fast (**38.1 ms** on `28m`, **122.4 ms** on `123m`, only **15.7%–17.9%** of wall time).
Over **75% of the entire SIMD wall time** (**182 ms / 242 ms** on `28m`, **516 ms / 685 ms** on `123m`) is spent in **single-threaded scalar fallbacks** in `Add`/`Mul`/`NotEqual` (**29.7%**), `Slice` (**24.0%**), and `Transpose` (**21.7%**)! Because 75% of wall time runs on **1 core**, overall average CPU utilization is pulled down to ~3.1–3.6 cores.

---

## 3. Root Causes of the Scheduler Regression (PRs #45 & #46)

### Cause 1: Defaulting to `OpsExecutionSequential` instead of `OpsExecutionDynamic`
- In `compute/internal/gobackend/gobackend.go`, PR #45 changed `newDefaultBackend()` from `OpsExecutionDynamic` (which uses `OpsExecutionParallel` whenever `NumLiveExecutions <= 1`) to `OpsExecutionSequential`.
- Because 75% of Timestral's SIMD wall time is spent in single-threaded ops (`Slice`, `Transpose`, `Add`, `Mul`, `NotEqual`) across independent branches of the graph (multi-scale branches, Q/K/V slices, RoPE sin/cos branches, gated MLP branches), `OpsExecutionSequential` serializes all of those 1-core ops.
- **Verification with `-backend="go:ops_parallel"` (without any code changes)**:
  - `model-28m` SIMD: **0.258s $\to$ 0.171s** (5.4 cores; faster than Oct 29's 0.187s!)
  - `model-123m` SIMD: **0.678s $\to$ 0.503s** (6.0 cores)
  - `model-28m` NoSIMD: **0.707s $\to$ 0.573s** (16.4 cores)
  - `model-123m` NoSIMD: **2.332s $\to$ 2.032s** (19.6 cores)

### Cause 2: Coarse Chunking & Wave Quantization in MatMul `AppendWorkItems` (`compute/internal/gobackend/dot/matmul/nosimd.go`)
- Before PR #45, `FeedWorkItems` targeted **`2 * maxWorkers`** work items (`targetWorkersPerBatch := (2*maxWorkers + batchSize - 1) / batchSize`) and rounded `lhsSplitSize` / `rhsSplitSize` **down** to multiples of the kernel register block (`LHSL1KernelRows` / `RHSL1KernelCols`).
- PR #45/#46 changed `targetWorkersPerBatch` to **`1 * maxWorkers`** (`(maxWorkers + batchSize - 1) / batchSize`) and changed `lhsSplitSize` / `rhsSplitSize` to round **up** (`((lhsSplitSize + mr - 1) / mr) * mr`).
- On `GOMAXPROCS=32`, this causes severe core under-subscription and tail-wave stalls on Timestral's exact shapes:
  1. **`M = 810` (`[2, 405, 640] × [640, 3424]`)**: `lhsSplitSize = ceil(810 / 32) = 26`, rounded **up** to a multiple of `LHSL1KernelRows = 8` (AVX-512) gives `32`. That produces only `ceil(810 / 32) = 26` work items — leaving **6 out of 32 cores completely idle** for the entire matmul!
  2. **`batchSize = 20` (`[2, 10, 378, 378] × [2, 10, 378, 64]`, 140 ms in 123M `nosimd`)**: `targetWorkersPerBatch = ceil(32 / 20) = 2`, producing `20 × 2 = 40` work items. Wave 1 runs 32 items on 32 cores; Wave 2 runs the remaining **8 items on 8 cores while 24 cores sit idle** (`40 / 64 = 62.5%` efficiency)!
  3. **`batchSize = 12` (in `28m`)**: `targetWorkersPerBatch = ceil(32 / 12) = 3`, producing `12 × 3 = 36` work items — Wave 2 runs **4 items on 4 cores while 28 cores sit idle** (`36 / 64 = 56.2%` efficiency)!
- Because `largeNoSIMDGeneric` and `avx512LargeFloat32` now allocate scratch buffers **once per worker goroutine** and pull from `workItems` via an atomic index (`nextItem.Add(1)`), targeting `2 * maxWorkers` to `4 * maxWorkers` work items adds **zero** buffer allocation or goroutine creation overhead while enabling dynamic load balancing and eliminating tail waves.

### Cause 3: `Choose2DSplit` Over-Penalizes 1D Row Splits (`candN == 1`) in `nosimd`
- In `nosimd_router.go`, `LHSPanelCrossSize = 2` and `RHSPanelCrossSize = 512`.
- In `Choose2DSplit` (`nosimd.go:244-254`), whenever `rowChunk > targetRow` (`2`), it assumes LHS cannot stay in cache (`lhsPanelFits = false`) and multiplies `lhsPackCost` by `nPanels = ceil(colChunk / 512)` plus a quadratic `colPenalty`.
- However, `largeNoSIMDGeneric` (`nosimd_large.go:63-64`) caches up to `maxCachedLHSPanels = 16` panels (`canCacheAllLHS` is `true` whenever `rowChunk <= 32`) so LHS is packed **only once** even when `candN == 1`! Because `Choose2DSplit` ignores `canCacheAllLHS` and removed the `candN == 1` preference, it forces 2D splits (`candN = 2, 4, 8`) where multiple workers redundantly pack the same LHS rows and write to adjacent segments of the same output rows.

---

## 4. Detailed Optimization Plan

### Step 1: Scheduler & MatMul Work-Partitioning Fixes (`compute/internal/gobackend/dot/matmul`)
1. **MatMul `AppendWorkItems` (`nosimd.go`)**:
   - Target `2 * maxWorkers` (or `4 * maxWorkers`) work items (`targetWorkersPerBatch := (4*maxWorkers + batchSize - 1) / batchSize`) so batched matmuls (`batchSize = 12, 20`) and single matmuls have enough granularity for dynamic atomic work stealing without tail-wave idle cores.
   - Round `lhsSplitSize` / `rhsSplitSize` **down** to a multiple of `LHSL1KernelRows` / `RHSL1KernelCols` (clamped to `>= mr` / `>= nr`), instead of rounding up.
2. **MatMul `Choose2DSplit` (`nosimd.go`)**:
   - Account for `canCacheAllLHS` (`rowChunk <= 16 * targetRow`) when determining whether LHS is repacked across `nPanels`.
   - Restore preference for `candN == 1` when `rowChunk` fits in cached LHS panels so workers do not redundantly pack LHS or share output rows.
3. **Evaluate `OpsExecutionDynamic` vs `OpsExecutionSequential` (`gobackend.go`)**:
   - Measure both after intra-op parallelization is added in Steps 2–4.

### Step 2: Optimize `Slice` with Contiguous Block Copies & Parallelization (`compute/internal/gobackend/ops/slice.go`)
- **Problem**: `execSliceGeneric` copies 1 element at a time on 1 thread, running an inner `for axis := rank - 1; axis >= 0; axis--` loop per element, even though every `Slice` in Timestral has `strides = [1, ..., 1]` with contiguous inner blocks of `32`, `1024`, `1712`, `241,920`, or `518,400` floats.
- **Solution**:
  - Collapse all trailing axes that have `stride == 1` (and full dimension extent for axes after the innermost sliced axis) into a contiguous `blockSize` copied via `copy()`.
  - Add fast paths for:
    1. **1D contiguous slice**: single `copy()` (or parallel chunked `copy()` if very large).
    2. **2D strided-row block copy (`[Outer, Inner]`)**: loop over `outerIdx` copying `blockSize` elements per row, parallelized via `backend.Scheduler.ParallelizeChunks` when `totalSize >= 32768`.
    3. **General N-D outer coordinates + contiguous inner `blockSize`**: advance outer coordinates once per `blockSize` elements instead of once per element, parallelized when `totalSize >= 32768`.
- **Expected Impact**: `164.5 ms -> ~5 ms` in `123m`; `59.3 ms -> ~2 ms` in `28m`.

### Step 3: Optimize `Transpose` with Axis Collapsing + Fast Batched 2D (`[0, 2, 1]`) & 4D (`[1, 0, 3, 2]`) Kernels (`compute/internal/gobackend/ops/transpose.go`)
- **Problem**:
  - `perm = [0, 2, 1]` (`[20, 405, 64]`, `[20, 378, 64]`, `[4050, 2, 64]` — **48.8 ms** in `123m`) is a batched 2D transpose (`[B, H, W] -> [B, W, H]`), which misses `dispatchFastTranspose` because only `rank == 2` and `rank == 4` are handled.
  - `perm = [2, 0, 1, 4, 3]` (`[2, 405, 3, 64, 10]` and `[405, 2, 3, 64, 10]` — **87.6 ms** in `123m`) has consecutive input axes `(0, 1)` that stay adjacent in `perm`, so it is really a 4D transpose `[810, 3, 64, 10]` with `perm = [1, 0, 3, 2]`.
- **Solution**:
  - Add a **canonicalize/collapse pre-pass** (`simplifyTranspose(dims, perm)`) that:
    1. Drops any axis with `dim == 1`.
    2. Merges any two adjacent axes in `perm` where `perm[i+1] == perm[i] + 1` (combining their dimensions `dims[perm[i]] * dims[perm[i+1]]`).
  - Add parallel tiled fast paths in `dispatchFastTranspose` for:
    1. **Rank 3 `perm = [0, 2, 1]`** (`[B, H, W] -> [B, W, H]`): uses the existing 2D tiled / SIMD transpose per batch `b`, parallelized across batches (or `(b, tileRow)`).
    2. **Rank 4 `perm = [1, 0, 3, 2]`** (`[D0, D1, D2, D3] -> [D1, D0, D3, D2]`): parallelized across `(d0, d1)` pairs, calling the 2D transpose kernel on each `[D2, D3]` plane.
- **Expected Impact**: `148.4 ms -> ~6 ms` in `123m`; `52.3 ms -> ~2 ms` in `28m`.

### Step 4: Add 3-Part Middle Broadcast (`BroadcastMidRHS` / `BroadcastMidLHS`) + Parallelize Binary & Comparison Ops (`compute/internal/gobackend/broadcast.go`, `ops/binaryops_simd.go`, `ops/gen_exec_binary.go`)
- **Problem**:
  - `Add((Float32)[2, 10, 378, 378], (Float32)[2, 1, 378, 378])` (**72.2 ms** in `123m`) and `Mul((Float32)[2, 405, 10, 64], (Float32)[2, 405, 1, 64])` (**62.4 ms** in `123m`) have shape pattern `[Outer, Mid, Inner]` op `[Outer, 1, Inner]`.
  - `DetermineBroadcastConfig` only recognizes 2-part broadcasts (`BroadcastPrefix*` and `BroadcastSuffix*`), so 3-part `[Outer, Mid, Inner]` falls into `BroadcastGeneral`, completely bypassing SIMD and running a single-threaded scalar `ZipIterator`!
  - `NotEqual((Float32)[2, 10, 378, 378], [2, 10, 378, 378])` (**30.7 ms** in `123m`) and large `Add`/`Mul` run on a single thread.
- **Solution**:
  - Add `BroadcastMidRHS` (`[Outer, Mid, Inner]` op `[Outer, 1, Inner]`) and `BroadcastMidLHS` (`[Outer, 1, Inner]` op `[Outer, Mid, Inner]`) to `BroadcastType` and `DetermineBroadcastConfig` (after squeezing 1-dims where both sides are 1, if exactly one contiguous range of dims has `dim == 1` on one side and `dim > 1` on the other, factor into `Outer`, `Mid`, `Inner`).
  - Handle `BroadcastMidRHS` and `BroadcastMidLHS` in `binaryops_simd.go` (for each `o` in `0..Outer`, broadcast the `[1, Inner]` slice across `m` in `0..Mid` using the existing contiguous SIMD vector loop `fn(lhsSlice, rhsInnerSlice, outSlice)`) and in `gen_exec_binary.go`.
  - Parallelize binary and comparison ops (`Add`, `Mul`, `Sub`, `Div`, `NotEqual`, `Equal`, etc.) across `Scheduler.ParallelizeChunks` when `totalElem >= 32768`.
- **Expected Impact**: `203.5 ms -> ~8 ms` in `123m` SIMD (`251.7 ms -> ~15 ms` in `123m` NoSIMD).

### Step 5: Optimize `Where` In-Place Aliasing & Trailing Broadcast (`compute/internal/gobackend/ops/where.go`)
- **Problem**:
  - When `Where(cond, scalarTrue, tensorFalse)` reuses `tensorFalse` as `output` in-place (`&out[0] == &onFalseFlat[0]`), `parallelWhere` still writes `out[i] = onFalseFlat[i]` for every `false` element (**52.5 ms** in `123m` NoSIMD).
  - When `onTrue` or `onFalse` has trailing dimension `1` (`Where([2, 864, 21], [2, 864, 1], [2, 864, 21])`), `newTernaryZipIterator` has `trailingLen = 1` and recomputes multi-dimensional strides on every single element.
- **Solution**:
  - Check if `out` aliases `onFalseFlat` (or `onTrueFlat`) in `parallelWhere` and skip writing the unchanged branch (`if cond[i] { out[i] = onTrueVal }`).
  - Add a fast path for `[Outer, Inner]` with `[Outer, 1]` broadcast on `onTrue` or `onFalse`.
- **Expected Impact**: `63.0 ms -> ~6 ms` in `123m` NoSIMD.

### Step 6: Optimize Remaining Non-MatMul Operations (`Concatenate`, `ReduceSum`/`ReduceMax`/`ReduceMin`, `Neg`/Unary Ops, `FusedSoftmax`, `FusedActivation`, and `Slice` Sub-Block Chunking)
- **Problem**:
  After Steps 1–5, per-op profiling of `model-123m` revealed the next tier of non-MatMul bottlenecks:
  1. **`Concatenate`** (`7.94 ms` SIMD / `9.22 ms` NoSIMD across 98 calls, e.g. `40× Concatenate([2, 405, 10, 32], [2, 405, 10, 32]) -> [2, 405, 10, 64]` and `21× Concatenate([2, 378, 640], [2, 27, 640]) -> [2, 405, 640]`):
     - `execConcatenateGeneric` ran single-threaded and iterated over `inputs` on the outer loop and `outerSize` (`8,100`) on the inner loop, jumping across non-contiguous output addresses (`outIdx += outputAxisBaseDim`) on every `32`-element copy and sweeping the `2 MB` output buffer multiple times.
  2. **`ReduceSum` / `ReduceMax` / `ReduceMin` in `nosimd`** (`10.77 ms` NoSIMD across 101 calls, dominated by `46× ReduceSum([2, 405, 640] -> [2, 405])` at `10.72 ms` = `233.1 µs/call` for RMSNorm):
     - While SIMD had a parallel fast path in `reduce_simd.go`, the generic fallback in `reduce.go` ran strictly single-threaded with a single scalar accumulator chain (`outputFlat[outIdx] += operandFlat[i]`).
  3. **`Neg` & Unary Operations** (`5.31 ms` NoSIMD / `1.23 ms` SIMD across 41 calls, e.g. `40× Neg([2, 405, 10, 32])`):
     - Both `unaryops.go` and `unaryops_simd.go` executed strictly on a single core regardless of tensor size.
  4. **`FusedSoftmax` & `FusedActivation` In-Place Reuse + Small-Axis Fast Path** (`11.72 ms` + `4.92 ms` SIMD / `22.19 ms` + `19.42 ms` NoSIMD):
     - Neither `FusedSoftmax` nor `FusedActivation` checked `inputsOwned[0]` to reuse the input buffer in-place (`ReuseInputOrGetBuffer`), allocating a fresh `11.4 MB` (`[2, 10, 378, 378]`) or `5.5 MB` (`[2, 405, 1712]`) buffer on every call.
     - `20× FusedSoftmax([405, 10, 2, 2], axis=3)` (`1.90 ms` in SIMD = `95.2 µs/call`) had `softmaxDim = 2`, causing `fusedSoftmaxRowFloat32` to pad 14 of 16 AVX-512 lanes with `-Inf` and run full 16-lane polynomial `ExpFloat32` on 8,100 2-element rows.
  5. **Coarse `Slice` Outer-Row Chunking (`count = 2`)**:
     - `60× Slice([2, 405, 10, 64] -> [2, 378, 10, 64])` (`4.54 ms` SIMD / `5.77 ms` NoSIMD) has `numOuter = 1` with `count = 2` outer rows (`o = 0, 1`) and `blockSize = 241,920` floats (`968 KB` per row). Parallelizing only across `count = 2` could use at most 2 worker threads.
- **Solution**:
  1. **`Concatenate` (`compute/internal/gobackend/ops/concatenate.go`)**: Invert loop order so `outerIdx` is the outer loop (writing all inputs sequentially into contiguous output memory per outer row) and parallelize across `outerSize` via `backend.Scheduler.ParallelizeChunks` when `totalElements >= 65536`.
  2. **`Reduce` (`compute/internal/gobackend/ops/reduce.go`)**: Attach `Backend` to `ReduceOutputIterator`, parallelize `ReduceTrailing` and `ReduceMiddle` across `A` when `len(operandFlat) >= 16384 && A > 1`, and use 4-way accumulator unrolling (`s0, s1, s2, s3`) in `execReduceSumGeneric` to break scalar FP addition latency chains.
  3. **Unary Ops (`compute/internal/gobackend/ops/unaryops.go`, `unaryops_simd.go`)**: Parallelize unary ops across `backend.Scheduler.ParallelizeChunks` when `n >= 65536` (`chunkSize = 32768`).
  4. **`FusedSoftmax` & `FusedActivation` (`compute/internal/gobackend/fusedops/softmax.go`, `softmax_simd.go`, `activation.go`)**: Use `backend.ReuseInputOrGetBuffer(0, node.Shape, inputBuffers, inputsOwned)` for in-place execution when `input` is owned, and add a scalar `n < 8` fast path in `fusedSoftmaxRowFloat32` to avoid 16-lane SIMD padding overhead on tiny `softmaxDim` (such as `n = 2`).
  5. **`Slice` Sub-Block Chunking (`compute/internal/gobackend/ops/slice.go`)**: When `numOuter == 1 && count < 8 && blockSize >= 65536`, split each large `blockSize` row into `32,768`-element sub-blocks so `count = 2, blockSize = 241,920` distributes 16 sub-block copy tasks across all worker cores.
- **Expected Impact**: Reduce `Concatenate`, `ReduceSum`, `Neg`, `FusedSoftmax`, and `Slice` overhead by `~20 ms` in `123m` NoSIMD and `~7–10 ms` in `123m` SIMD.

---

## 5. Progress & Benchmark Log

### Step 1a: Default to `OpsExecutionDynamic` (`compute/internal/gobackend/gobackend.go`)

| Obs | Model | Run | Wall (Before) | Wall (Now) | Speedup | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5 | 2.332s | **2.038s** | 1.14× | 39.267s | 19.3 | 5.1e-05 | 2.03 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5 | 0.678s | **0.507s** | 1.34× | 2.837s | 5.6 | 1.5e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5 | 0.707s | **0.594s** | 1.19× | 9.268s | 15.6 | 3.8e-05 | 0.88 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5 | 0.258s | **0.171s** | 1.51× | 0.653s | 3.8 | 3.6e-05 | 0.77 GB |

### Step 1b: Finer-Grained MatMul & Scheduler Chunking (Causes 2 & 3)

Changes made:
1. **`Choose2DSplit` (`compute/internal/gobackend/dot/matmul/nosimd.go`) & `lhsAllocPanels` (`nosimd_large.go`)**:
   - Increased per-worker LHS panel cache capacity in `nosimd_large.go` from 16 to 64 panels (`128` rows = `256 KB`, since `NoSIMDParams.LHSPanelCrossSize = 2`).
   - Updated `Choose2DSplit` to account for `maxCachedLHSRows` (`max(16*LHSPanelCrossSize, 128)`) and restored the `-200.0` preference for 1D row splits (`candN == 1`) when `rowChunk >= targetRow`, avoiding redundant LHS re-packing across column workers.
2. **`AppendWorkItems` (`compute/internal/gobackend/dot/matmul/nosimd.go`)**:
   - Restored `targetWorkersPerBatch := (2*maxWorkers + batchSize - 1) / batchSize` so dynamic atomic work pulling can smooth out tail waves when `batchSize` does not divide `maxWorkers` (e.g. `batchSize = 12, 20`) or when concurrent inter-op branches occupy 1–2 cores.
   - Replaced uniform round-up `lhsSplitSize`/`rhsSplitSize` (which created only 26 work items for `M = 810` and 20 items for `M = 320`, starving 6–12 cores) with proportional register-block (`mr`/`nr`) distribution (`((m*totalMBlocks + actualM - 1) / actualM) * mr`).
3. **`ParallelizeChunks` & `SaturateUpTo` (`compute/internal/gobackend/scheduler/scheduler.go`)**:
   - Updated `ParallelizeChunks` to target `2 * numWorkers` chunks with proportional ceiling-first boundaries instead of uniform `chunkSize = ceil(totalItems / numChunks)` (which left 12 of 32 cores idle when `totalItems = 40`).
   - Replaced the post-loop `intraOpQueue` drain in `RunIntraOpTasks`, `ParallelizeChunks`, and `SaturateUpTo` (which could pop unrelated `SubmitInterOp` branch tasks while waiting for in-flight helpers) with atomic `unstarted` helper cancellation.

#### Timestral Results (`run 5`, with best run in parentheses):

| Obs | Model | Run | Wall (Step 1a) | Wall (Step 1b) | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5 | 2.038s | **2.004s** (min 1.947s) | 39.246s | 19.6 | 5.1e-05 | 2.11 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5 | 0.507s | **0.510s** (min 0.484s) | 3.116s | 6.1 | 2.3e-05 | 2.12 GB |
| amd9950x3d/nosimd | model-28m/ | run 5 | 0.594s | **0.545s** (min 0.531s) | 9.420s | 17.3 | 3.8e-05 | 0.86 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5 | 0.171s | **0.181s** (min 0.169s) | 0.824s | 4.5 | 3.8e-05 | 0.83 GB |

#### `ka_microbench` Results (Before vs After Step 1b):

| Mode | Test | Mean ms/call (Before) | Mean ms/call (After) | Texts/s (Before) | Texts/s (After) | Cores Busy (Before $\to$ After) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `nosimd` | `-batch 16 -words 20 -iters 30` | 97.22 ms | **71.99 ms** | 164.6 | **222.2 (+35%)** | 14.7 $\to$ 20.3 |
| `nosimd` | `-batch 8 -words 100 -iters 10` | 200.35 ms | **172.05 ms** | 39.9 | **46.5 (+16%)** | 17.6 $\to$ 20.6 |
| `nosimd` | `-batch 1 -words 20 -iters 60` | 8.34 ms | **8.03 ms** | 119.9 | **124.5 (+4%)** | 11.3 $\to$ 11.9 |
| `simd` | `-batch 16 -words 20 -iters 30` | 10.71 ms | **10.66 ms** | 1493.6 | **1501.5** | 6.7 $\to$ 7.3 |
| `simd` | `-batch 8 -words 100 -iters 10` | 30.41 ms | **29.96 ms** | 263.1 | **267.0** | 7.0 $\to$ 7.6 |
| `simd` | `-batch 1 -words 20 -iters 60` | 3.31 ms | **3.28 ms** | 301.8 | **304.5** | 1.7 $\to$ 1.6 |

### Step 2: Optimize `Slice` with Contiguous Block Copies, Outer Axis Merging & Parallelization (`compute/internal/gobackend/ops/slice.go`)

- **Note on negative strides / `Reverse` (issue #472)**: Profiled `Reverse` across Timestral — it is called only **2 times** per inference pass, taking **0.14–0.18 ms** total (**< 0.07%** of wall time). All 311 `Slice` calls in `28m` and 511 `Slice` calls in `123m` have positive `strides = [1, ..., 1]`.
- **Implementation**:
  1. Collapse trailing dimensions with `stride == 1` into a single contiguous `blockSize` copied via `copy()`.
  2. Drop size-1 outer axes and merge adjacent outer axes where `step[prev] == dim[curr] * step[curr]` (using stack-allocated `[8]int` buffers to eliminate per-call heap allocations). This reduces 3D/4D slices like `[2, 405, 3424] -> [2, 405, 1712]` and `[2, 405, 10, 64] -> [2, 405, 10, 32]` to a single 1D outer loop of `copy()` calls (`numOuter == 1`).
  3. Parallelize across `backend.Scheduler.ParallelizeChunks` when `len(outputFlat) >= 65536`.

#### Timestral Results (`run 5`, with best run in parentheses):

| Obs | Model | Run | Wall (Step 1b) | Wall (Step 2) | Total Speedup vs Baseline | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5 | 2.004s | **1.926s** (min 1.881s) | 1.21× | 39.643s | 20.6 | 5.1e-05 | 2.13 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5 | 0.510s | **0.452s** (min 0.452s) | 1.50× | 3.122s | 6.9 | 2.3e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5 | 0.545s | **0.521s** (min 0.521s) | 1.36× | 9.307s | 17.9 | 3.8e-05 | 0.81 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5 | 0.181s | **0.156s** (min 0.144s) | 1.65× | 0.815s | 5.2 | 3.8e-05 | 0.80 GB |

### Step 3: Optimize `Transpose` with Canonical Axis Simplification & Parallel `3D_021` / `4D_1032` Fast Paths (`compute/internal/gobackend/ops/transpose.go`)

- **Implementation**:
  1. Added `simplifyTranspose(dims, permutations)` (using stack-allocated `[8]int` buffers) which:
     - Drops size-1 axes and remaps remaining axis indices.
     - Repeatedly merges consecutive axes where `sPerm[i+1] == sPerm[i] + 1` into a single combined axis (`sDims[a] *= sDims[a+1]`).
     - This automatically reduces:
       - Any identity permutation to rank $\le 1$.
       - 5D `[2, 10, 15, 27, 64]` `perm=[0, 2, 1, 3, 4]` $\to$ 4D `[2, 10, 15, 1728]` `perm=[0, 2, 1, 3]` (routing it to the parallel `transpose4D_0213` kernel instead of the single-threaded `transposeTrailing` iterator).
       - 5D `[2, 405, 3, 64, 10]` and `[405, 2, 3, 64, 10]` `perm=[2, 0, 1, 4, 3]` $\to$ 4D `[810, 3, 64, 10]` `perm=[1, 0, 3, 2]`.
  2. Added `dispatchTranspose3D_021` / `transpose3D_021` for batched 2D matrix transposes `[B, H, W] -> [B, W, H]` (e.g. `[20, 405, 64]`, `[20, 378, 64]`, `[4050, 2, 64]`), parallelized across `B` when `B*H*W > 32768`.
  3. Added `dispatchTranspose4D_1032` / `transpose4D_1032` for simultaneous outer+inner 2D transposes `[D0, D1, H, W] -> [D1, D0, W, H]`, parallelized in contiguous output `(d1, d0)` order across `backend.Scheduler.ParallelizeChunks`.
  4. Added `transposeSlice2D` helper that uses a direct L1-resident row loop when `H*W <= 1024` (eliminating tiling loop overhead on `[64, 10]`, `[32, 8]`, `[2, 64]`) and 32×32 cache tiling for larger matrices.

#### Timestral Results (`run 5`, with best run in parentheses):

| Obs | Model | Run | Wall (Baseline) | Wall (Step 2) | Wall (Step 3) | Total Speedup vs Baseline | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5 | 2.332s | 1.926s | **1.794s** (min 1.794s) | **1.30×** | 39.832s | 22.2 | 5.1e-05 | 2.06 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5 | 0.678s | 0.452s | **0.322s** (min 0.322s) | **2.11×** | 3.182s | 9.9 | 2.3e-05 | 2.08 GB |
| amd9950x3d/nosimd | model-28m/ | run 5 | 0.707s | 0.521s | **0.486s** (min 0.480s) | **1.45×** | 9.250s | 19.0 | 3.8e-05 | 0.84 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5 | 0.258s | 0.156s | **0.110s** (min 0.110s) | **2.35×** | 0.704s | 6.4 | 3.8e-05 | 0.82 GB |

### Step 4: Add 3-Part Middle Broadcast (`BroadcastMidRHS` / `BroadcastMidLHS`) & Parallelize Binary / Comparison Ops (`compute/internal/gobackend/broadcast.go`, `ops/binaryops_simd.go`, `internal/cmd/gobackend_ops_generator/exec_binary.go`)

- **Implementation**:
  1. Refactored `DetermineBroadcastConfig` in `compute/internal/gobackend/broadcast.go` to classify axes into collapsed mode segments (`modeMatch`, `modeBcastRHS`, `modeBcastLHS`), automatically skipping neutral size-1 axes (`tgtDim == 1`) and merging consecutive axes with identical modes.
  2. Added `BroadcastMidRHS` (`[A, B, C]` op `[A, 1, C]`) and `BroadcastMidLHS` (`[A, 1, C]` op `[A, B, C]`) to `BroadcastPattern` and `BroadcastConfig`:
     - `Add((Float32)[2, 10, 378, 378], (Float32)[2, 1, 378, 378])` is now classified as `BroadcastMidRHS{A: 2, B: 10, C: 142884}` instead of falling back to the single-threaded scalar `ZipIterator`.
     - `Mul((Float32)[2, 405, 10, 64], (Float32)[2, 405, 1, 64])` is now classified as `BroadcastMidRHS{A: 810, B: 10, C: 64}` and dispatches each `a` slice through the register-cached `simdV2V1MulFloat32` kernel.
  3. Added multi-core parallelization via `backend.Scheduler.ParallelizeChunks` when `len(output) >= 65536` in both `dispatchBinarySIMD` (`binaryops_simd.go`) and the generated generic/comparison executors (`exec_binary.go` -> `gen_exec_binary.go`).

#### Timestral Results (`run 5`, with best run in parentheses):

| Obs | Model | Run | Wall (Baseline) | Wall (Step 3) | Wall (Step 4) | Total Speedup vs Baseline | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | run 5 | 2.332s | 1.794s | **1.658s** (min 1.658s) | **1.41×** | 39.458s | 23.8 | 5.1e-05 | 2.08 GB |
| amd9950x3d/AVX512 | model-123m/ | run 5 | 0.678s | 0.322s | **0.232s** (min 0.227s) | **2.92×** (2.99× min) | 3.395s | 14.6 | 2.3e-05 | 2.04 GB |
| amd9950x3d/nosimd | model-28m/ | run 5 | 0.707s | 0.486s | **0.425s** (min 0.421s) | **1.66×** | 9.306s | 21.9 | 3.8e-05 | 0.82 GB |
| amd9950x3d/AVX512 | model-28m/ | run 5 | 0.258s | 0.110s | **0.079s** (min 0.079s) | **3.27×** | 0.819s | 10.3 | 3.8e-05 | 0.79 GB |

### Step 5: Optimize `Where` with 2D-Collapsible Broadcasts, Parallelization & In-Place Aliasing (`compute/internal/gobackend/ops/where.go`)

- **Key Findings**:
  1. **Trailing-broadcast `Where` bottleneck (`trailingLen == 1`)**:
     In Timestral, `Where((Bool)[2, 10, 15, 27, 64], (Float32)[2, 10, 15, 27, 1], (Float32)[2, 10, 15, 27, 64])` (20 calls in `123m`, `518,400` elements/call = `10.37M` elements) and `Where((Bool)[2, 864, 21], (Float32)[2, 864, 1], (Float32)[2, 864, 21])` have a broadcast on the last axis (`64` or `21`). Previously, `ternaryZipIterator` stopped at `splitAxis = rank - 1` (`trailingLen = 1`), executing single-threaded, computing `3 * rank` stride multiplications per element, and in SIMD mode invoking `whereFloat32Arch` with `count = 1` (`518,400` assembly calls per op!).
  2. **Redundant writes on in-place buffer reuse (`output == onFalse` / `output == onTrue`)**:
     In `execWhere`, when `onFalse` (or `onTrue`) matches `outputShape` and is owned (`inputsOwned[2]`), `output` aliases `onFalse` in-place (which happens on every attention mask `Where(mask, -inf, scores)`). Yet the scalar loop wrote `out[i] = onFalseFlat[i]` on every unmasked element, dirtying cache lines unnecessarily.
- **Implementation**:
  1. Added `determineWhere2DConfig(condShape, trueShape, falseShape, tgtShape)` which classifies each non-trivial axis by its 3-bit broadcast mask `(condBcast, trueBcast, falseBcast)` and collapses consecutive axes with identical masks into at most 2 segments (`[outerSize, innerSize]`).
  2. Within each outer row `r` of a 2-segment `[outerSize, innerSize]` broadcast, any operand broadcast along `innerSize` (such as `[2, 10, 15, 27, 1]`) acts as a per-row scalar `flat[r : r+1]` across the contiguous `innerSize` slice (`64` elements), allowing `whereRow` to invoke SIMD `whereFloat32Arch` (e.g., `whereSVFloat32AVX2Asm`) on full 64-element rows and parallelize across `outerSize` via `backend.Scheduler.ParallelizeChunks`.
  3. Added in-place alias fast paths (`outAliasesFalse` / `outAliasesTrue`) in `whereRow` so scalar execution only writes elements that actually change (`if cond[i] { out[i] = tVal }`).
  4. Unified `dispatchWhereParallel` into `execWhereGeneric` via `whereDTypeMap`.

#### Timestral Results (`best` of 5 timed runs):

| Obs | Model | Run | Wall (Baseline) | Wall (Step 4) | Wall (Step 5) | Total Speedup vs Baseline | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | best | 2.332s | 1.658s | **1.601s** | **1.46×** | 39.678s | 24.8 | 5.1e-05 | 2.03 GB |
| amd9950x3d/AVX512 | model-123m/ | best | 0.678s | 0.227s | **0.220s** | **3.08×** | 3.538s | 16.1 | 2.3e-05 | 1.93 GB |
| amd9950x3d/nosimd | model-28m/ | best | 0.707s | 0.421s | **0.395s** | **1.79×** | 9.224s | 23.3 | 3.8e-05 | 0.71 GB |
| amd9950x3d/AVX512 | model-28m/ | best | 0.258s | 0.072s | **0.065s** | **3.97×** | 0.578s | 8.9 | 3.8e-05 | 0.71 GB |

### Step 6: Optimize `Concatenate`, `ReduceSum`/`ReduceMax`/`ReduceMin`, `Neg`/Unary Ops, `FusedSoftmax`/`FusedActivation` In-Place Reuse, and `Slice` Sub-Block Chunking

- **Key Findings**:
  1. **Dual-CCD Thresholding Trade-off (`65,536` vs `32,768` Elements)**:
     Testing a lower intra-op parallelization threshold (`32,768` elements with `16,384` chunk size) slightly improved `model-123m`, **but** increased CPU Time on `model-28m` AVX-512 from `0.578s` to `0.940s` (`0.065s -> 0.075s` wall). Under `OpsExecutionDynamic`, independent graph branches already run concurrently on `model-28m`, so spawning intra-op tasks on `32K`-element SIMD tensors causes cross-CCD cache coherence traffic on the dual-CCD Ryzen 9 9950X3D. Keeping the activation threshold at `65,536` elements with `32,768`-element chunks avoids over-fragmenting `model-28m` while providing 16–32 parallel chunks for `518,400`-element and `2,857,680`-element tensors in `model-123m`.
  2. **Per-Op Speedups on `model-123m` (`GOMLX_OP_PROFILE=1`)**:
     - `ReduceSum` (101 calls): **10.77 ms $\to$ 2.39 ms** (**4.51×** faster in NoSIMD; `ReduceSum([2, 405, 640])` dropped from `233.1 µs/call` to `50.9 µs/call`; SIMD dropped from `2.45 ms` to `1.82 ms`).
     - `Neg` (41 calls): **5.31 ms $\to$ 1.74 ms** (**3.05×** faster in NoSIMD; `1.23 ms -> 1.11 ms` in SIMD).
     - `Concatenate` (98 calls): **9.22 ms $\to$ 6.07 ms** (**1.52×** faster in NoSIMD) and **7.94 ms $\to$ 5.74 ms** (**1.38×** faster in SIMD) by inverting the loop order (`outerIdx` outside, writing all inputs contiguously per outer row) and parallelizing across `outerSize`.
     - `FusedSoftmax` (45 calls): **11.72 ms $\to$ 9.89 ms** (**1.19×** faster in SIMD) via in-place buffer reuse (`ReuseInputOrGetBuffer`) and a scalar `n < 8` fast path in `fusedSoftmaxRowFloat32` (eliminating 16-lane `-Inf` padding and polynomial `ExpFloat32` on `FusedSoftmax([405, 10, 2, 2])`).
     - `Slice([2, 405, 10, 64] -> [2, 378, 10, 64])` (60 calls, `count = 2, blockSize = 241,920`): **5.77 ms $\to$ 4.21 ms** (**1.37×** faster in NoSIMD) and **4.54 ms $\to$ 3.80 ms** in SIMD by splitting large outer blocks into `32,768`-element sub-blocks when `count < 8`.

#### Timestral Results (`best` of 5 timed runs):

| Obs | Model | Run | Wall (Baseline) | Wall (Step 5) | Wall (Step 6) | Total Speedup vs Baseline | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amd9950x3d/nosimd | model-123m/ | best | 2.332s | 1.601s | **1.577s** | **1.48×** | 39.545s | 25.1 | 2.7e-05 | 1.98 GB |
| amd9950x3d/AVX512 | model-123m/ | best | 0.678s | 0.220s | **0.212s** | **3.20×** | 3.478s | 16.4 | 2.3e-05 | 1.90 GB |
| amd9950x3d/nosimd | model-28m/ | best | 0.707s | 0.395s | **0.405s** | **1.75×** | 9.198s | 22.7 | 3.4e-05 | 0.75 GB |
| amd9950x3d/AVX512 | model-28m/ | best | 0.258s | 0.065s | **0.068s** | **3.79×** | 0.705s | 10.4 | 3.8e-05 | 0.75 GB |







