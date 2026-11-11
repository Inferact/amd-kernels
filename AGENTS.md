# Agent instructions

Writing rules for code and docs in this repository. The kernel rules are in [README.md](README.md) ("Kernel rules").

- Minimize comments. Delete comments that restate the code; keep one only for a reason the code cannot show (a
  hardware hazard, a measured crossover, an ordering a caller relies on).
- No history in code, docstrings or READMEs: no superseded variants, dates or commit references. Git history and PR
  bodies hold them.
- Code adapted from another project keeps its attribution: the original copyright line in the file header and one
  line naming the source (project, file or PR, revision). That is attribution, not history.
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
