# Agent instructions

Writing rules for code and docs in this repository. The kernel rules are in [README.md](README.md) ("Kernel rules").

- Minimize comments. Delete comments that restate the code; keep one only for a reason the code cannot show (a
  hardware hazard, a measured crossover, an ordering a caller relies on).
- No history in code, docstrings or READMEs: no superseded variants, dates or commit references. Git history and PR
  bodies hold them.
- Code or ideas taken from another project (AITER, Triton, CK, FlashInfer, SGLang, a paper's code, any other) name
  their source. That is attribution, not history.
  - Copied or adapted code: after this repo's SPDX lines, a `# Portions adapted from` line with the project and the
    URL of the PR, the source file's path and revision when there is one, then the source file's SPDX license line
    and copyright line as they are (no license text).
  - The module docstring says in one sentence which functions, kernel or technique came from it. A borrowed idea
    without copied code gets this sentence only.

  ```python
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project

  # Portions adapted from ROCm/aiter (https://github.com/ROCm/aiter/pull/5750),
  # aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale_group32.py at e2d019f:
  # SPDX-License-Identifier: MIT
  # Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
  """MXFP8 GEMM on 32x32 block-scaled weights.

  The packed kernel and the in-launch split-K reduction (``_split_tile``, ``_sum_splits``) are adapted from the
  group32 GEMM in ROCm/aiter#5750.
  """
  ```
- A constant chosen by measurement gets at most one line: the result and where the sweep lives.
- Docstrings: a one-line summary; Google-style `Args:` / `Returns:` only where a parameter is not obvious from its name
  and type.
- A kernel package imports only itself (relative imports), torch, FlyDSL and AITER; add one line for it to
  [amd_kernels/README.md](amd_kernels/README.md).
- An entry README holds only what the code cannot show, in at most 60 lines and four sections: what it does
  (computation, shapes, dtypes, numerics contract), entry points (one line each: the dotted name and what a caller
  uses it for, no signatures or argument lists; one line on where the dispatch lives), performance (one table: graph
  replay, M -> us, vs the baseline), tests (the commands). Signatures and the shape -> kernel dispatch live in the
  code; a design fact the code cannot show (a hand-off protocol, a hazard, why a tile) goes into the module docstring
  of the kernel file it describes. A new result replaces its table.
