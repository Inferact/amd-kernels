# Agent instructions

Writing rules for code and docs in this repository. The kernel rules are in [README.md](README.md) ("Kernel rules").

- Minimize comments. Delete comments that restate the code; keep one only for a reason the code cannot show (a
  hardware hazard, a measured crossover, an ordering a caller relies on).
- No history in code, docstrings or READMEs: no superseded variants, dates or commit references. Git history and PR
  bodies hold them.
- Code or ideas taken from another project (AITER, Triton, CK, FlashInfer, SGLang, a paper's code, any other) follow
  that project's license and name it, as vllm-project/vllm#58659 does for AITER. That is attribution, not history.
  - Copied or adapted code: after this repo's SPDX lines, `# Portions adapted from <project> (<URL of the PR or
    file>):`, then the source's SPDX license line and its copyright line.
  - The module docstring names what came from where: the functions, kernel or technique, and the source PR or file
    at its revision. A borrowed idea without copied code gets this sentence only.
  - No license, or one this repo cannot take (GPL), means no copy. The PR body says what was adapted from where.
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
