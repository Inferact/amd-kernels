# amd-kernels

FlyDSL kernels for AMD Instinct MI355X (gfx950), as a Python package (`amd_kernels`) that model code calls directly.

Every kernel package and its entry points: [amd_kernels/README.md](amd_kernels/README.md).

## Purpose and status

The library gives vLLM day-0 kernel support for new models on AMD GPUs: the kernels a model needs on MI355X, written
and tuned when the model lands. Today it serves vLLM only, and its Python API follows what vLLM's model code needs, so
it may change without notice: do not depend on it directly. We plan to offer these kernels as an
[AITER plugin](https://github.com/ROCm/aiter/blob/main/aiter/plugin/README.md) later.

## Requirements

- MI355X, ROCm 7.2, Python 3.12, PyTorch for ROCm.
- [FlyDSL](https://github.com/ROCm/FlyDSL) >= 0.5.0 and [AITER](https://github.com/ROCm/aiter).

## Install and test

```bash
pip install --no-deps -e .                            # keeps your torch, AITER and FlyDSL
source env.sh                                         # checkout on PYTHONPATH, JIT caches in .cache/
python3 -m pytest tests/<category>/<name> -q
python3 benchmarks/<category>/<name>/<benchmark>.py   # CUDA graph replay over inputs larger than the LLC
```

## Kernel rules

### 1. AMD only

Kernels target AMD Instinct GPUs (today MI355X, gfx950). There is no platform-agnostic API or kernel registry; the
host code picks the variant by shape.

### 2. FlyDSL only

No Triton, Gluon, CuteDSL or HIP `.cu`; glue is torch ops or FlyDSL. FlyDSL covers what we would otherwise write in
HIP, Triton or Gluon: tile-level ops plus direct buffer loads, LDS DMA, MFMA and inline asm. One language gives one
set of tools, tests and recorded lessons. We accept longer and more complex kernel code for that.

### 3. The torch reference is the contract

Learned from [TokenSpeed kernels](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/README.md),
whose PyTorch references are the ground truth their kernels are tested against. One reference function per op, in the
kernel package's `reference.py`; tests and benchmarks import it and never write their own. Its docstring states what
"correct" means: where each rounding and quantization happens, the scale layout, the accumulate dtype, and the tolerance
the tests use (cosine ~1 and deterministic, not bitwise). Test shapes come from the served regime, including CUDA graph
padding rows.

### 4. Every variant compiles before serving

All variants a server can hit (every shape tier and config) compile during warm-up or CUDA graph capture. An entry
registered with `amd_kernels.warm_up.entry` warms itself up on its first call of each mode, compiling in worker
processes ([`warm_up.py`](amd_kernels/warm_up.py)); after a caller's warm-ups, a compile is reported with its call site
([`compile_watch.py`](amd_kernels/compile_watch.py)). A test walks
the batch sizes after a warm-up and asserts that nothing compiles.

### 5. One kernel per served shape is allowed

A kernel may be written for exactly the shapes it serves, and a new shape may get its own copy instead of another
branch in a shared kernel. The fastest instruction order and scheduling differ from shape to shape (tile, pipeline
depth, how MFMAs interleave with loads, epilogue, grid), so a copy per shape lets an agent push each one to its limit
without regressing the others. We accept the duplicated code for that; a review then touches one shape.

### 6. Names carry the computation and the shape

Never the deployment, the model release, an iteration number or where the code came from. Shape tokens name the
dimensions the kernel fixes (`h` hidden, `i` per-GPU MoE intermediate, `e` experts, `n` / `k` of a GEMM weight,
`ranks<n>` for multi-GPU kernels), e.g. `moe/mxfp4_h6144_i768_e128_top4`; the token count is a runtime argument.

### 7. Deterministic: no atomic adds into outputs

A kernel never accumulates its output with atomic adds. Split-K and expert partials are summed in a fixed order: by a
reduce kernel, by the last workgroup of a tile, or inside the next kernel (the all-reduce, the norm / RoPE / KV insert).
Atomics only count tickets and flags. A wrong result then reproduces run to run, which an atomic sum cannot promise.
We accept the reduce step that an atomic would skip for that.

## Layout

```text
amd_kernels/<category>/<name>/   kernels, reference.py, README.md
tests/<category>/<name>/         tests against reference.py
benchmarks/<category>/<name>/    graph-replay benchmarks
```

Writing rules for code and docs: [AGENTS.md](AGENTS.md).

## Acknowledgements

- [FlyDSL](https://github.com/ROCm/FlyDSL): every kernel here is written in it and compiled by it.
- [AITER](https://github.com/ROCm/aiter): the shared FlyDSL helpers in [`common/`](amd_kernels/common/) (buffer ops,
  launch helpers, the HIP runtime binding) come from AITER's FlyDSL kernels, as do the MXFP8 GEMM's preshuffled
  scaled-MFMA kernel and its scale layout; AITER's kernels are the baselines many benchmarks compare against.
- [TokenSpeed kernels](https://github.com/lightseekorg/tokenspeed) (`tokenspeed-kernel`): the kernel contract this
  library follows, a PyTorch reference as the ground truth that tests compare against, standalone graph-replay
  benchmarks and a check for compiles while serving.

## License

Apache-2.0, see [LICENSE](LICENSE).
