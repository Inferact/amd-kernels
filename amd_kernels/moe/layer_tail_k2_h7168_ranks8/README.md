# Layer tail: o_proj all-reduce + AttnRes + routed and shared MoE + up projection, h7168, eight ranks

## What it does
One decode layer's tail on each of the eight TP ranks of Kimi-K3.1, in one persistent launch (256 workgroups x 512
threads, all resident): `o_proj` (+ all-reduce) -> AttnRes (softmax mix over the block sources, RMSNorm) -> router
(sigmoid + bias, top-16) -> shared SiTU MLP and the routed MXFP4 experts (gate/up, down) -> latent all-reduce +
RMSNorm -> up projection -> all-reduce. Hidden 7168, latent 3584, 896 routed experts (each rank holds its TP shard),
rows 1..16 (graph batches 4, 6, 8, 12, 16). `core` bf16 [S, 1536] is o_proj's input, `prefix` bf16 [S, 7168] is
updated in place (restarted at the o_proj output on a block-write layer, `reset`), `blocks` bf16 [S, B, 7168] are the
AttnRes sources, the output is bf16 [S, 7168]. Weights are used as vLLM holds them (AITER-shuffled MXFP4 experts with
E8M0 scales; no copies, no requantization).

Numerics are those of vLLM's mono-kernel layer tail bit for bit (same accumulation order, same bf16 roundings);
`reference.py` is the pure-torch contract (cosine >= 0.9999 against the kernel on its own x; the picks may differ on a
near tie). Rank partials are summed in rank order, no atomic adds into outputs. Stages hand data over by tagged words
in a per-rank scratch and the peers' mailboxes (store + local poll, about 0.7 us a hop; no grid barrier); a wait that
gives up after 10 s sets a health word instead of hanging. The workgroup roles are in the module docstring of
`kernel.py`.

## Entry points
- `LayerTailRunner`: the per-rank state (scratch, epoch, health words, peer memory); collective, once per TP8 group.
- `LayerTailRunner.step_begin`: once a forward, before its first layer (captured in the graph).
- `LayerTailWeights`: one layer's weights, views of vLLM's tensors.
- `layer_tail`: one layer's tail (an `amd_kernels.warm_up` entry: the first call of a mode (nblocks, reset) compiles
  the served widths, launching nothing).
- `reference.layer_tail`: the contract.
No row-count dispatch: `host.py` builds one kernel per (rows, nblocks, reset).

## Performance
Graph replay of 4 layers over distinct weight sets (past the LLC), 8 ranks, slowest-rank median us a layer, baseline
vLLM's mono-kernel layer tail in the same harness; roofline = 1.6 us + bytes one rank reads / 6.4 TB/s (dense ~70 MB
+ picked experts x 2.2 MB).

| rows | experts picked | stock us | this kernel us | speedup | roofline us | % of roofline |
|---|---|---|---|---|---|---|
| 12 | uniform (168) | 118.1 | 108.5 | 1.09x | 73.8 | 68% |
| 6 | uniform (90) | 85.7 | 78.7 | 1.09x | 44.7 | 57% |
| 12 | 96 | 94.2 | 84.4 | 1.12x | 43.0 | 51% |
| 6 | 96 | 76.0 | 69.8 | 1.09x | 34.1 | 49% |

The expert stream runs at the nontemporal read ceiling (~6.8 TB/s in `ug`, ~6.1 in `down`); the rest is the dependent
chain (AttnRes and router 32 us before the first expert byte, 15 us after the last).

## Tests
```bash
python3 -m pytest tests/moe/layer_tail_k2_h7168_ranks8 -q   # test_layer_tail: 8 GPUs; test_warm_up: compile only
torchrun --nproc_per_node 8 benchmarks/moe/layer_tail_k2_h7168_ranks8/benchmark_layer_tail.py
```
