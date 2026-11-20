# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Both decoder-layer tails at two ranks (``allreduce_add_rmsnorm_ranks2``) against the sequence vLLM runs at TP2
without the entry, on 2 gfx950 GPUs:

- ``bf16`` (attention tail of a MoE layer): rows <= 80 AITER's one-stage fused all-reduce + Gemma norm (vLLM's
  ``use_1stage_fused_ar_rms`` at world 2), above AITER's custom all-reduce + the one-GPU ``gemma_add_rmsnorm``;
- ``mxfp8`` (attention tail of a dense layer): AITER's custom all-reduce + the one-GPU MXFP8 norm (codes tiled to 16
  rows up to 512 rows, row-major above);
- ``slots`` (MoE tail): the entry sums the rank's 4 weighted slots + shared expert itself; the baseline all-reduces one
  bf16 partial (its MoE combine is not counted) + the one-GPU MXFP8 norm.

Graph replay: every arm cycles through distinct inputs (and residuals) whose total exceeds 400 MB (beyond the 256 MB
MALL), captured in graphs of at most 256 calls; rounds alternate the arm order, each sample is the max over the 2
ranks of the per-call time, printed median [min-max] over rounds. ``--sweep`` times the entry's
(rows per batch, stagers, reducers, coded) candidates instead.

    python3 benchmarks/norm/gemma_add_rmsnorm_h6144/benchmark_ar_add_rmsnorm_ranks2.py [--kinds bf16,mxfp8,slots]
        [--rounds 7] [--json out.json] [--sweep] [rows ...]
"""

import argparse
import json
import statistics

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

EPS = 1e-6
H = 6144
ROWS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4095]
LLC_BYTES = 400 << 20
MAX_GRAPH_CALLS = 256
LINK_GBPS = 56.0  # measured per direction with both directions busy


def _sets(rows: int, kind: str) -> int:
    per_call = rows * H * 2 * (2 + (5 if kind == "slots" else 1))
    return max(8, min(2048, -(-LLC_BYTES // per_call)))


def _sweep_configs(rows):
    out = []
    for coded in (False, True):
        for rpb in (1, 2, 4, 8):
            nb = -(-rows // rpb)
            for nstage in (16, 32, 64, 128):
                for nred in (32, 64, 128, 256):
                    c = (rpb, min(nb, nstage), min(nb, nred), coded)
                    if c not in out and c[1] + c[2] <= 512:
                        out.append(c)
    return out


def _worker(rank, errors, rows_list, kinds, rounds, out_json, sweep):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm_ranks2 as arn
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm_ranks2 import host as h2

    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    ca = CustomAllreduce(group=group, device=dev, max_size=8192 * 1024 * 8 * 2)  # vLLM's MAX_SIZE
    assert not ca.disabled
    buffers = h2.SharedBuffers(rank, group, dev)
    staged = arn.staged_comm(rank, group, dev, buffers)
    slots_comm = arn.slots4_comm(rank, group, dev, buffers)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(39)) * 0.5).bfloat16().to(dev)
    staged.warm_up(w, EPS)
    slots_comm.warm_up(w, EPS)
    results = {}
    for rows in rows_list:
        for kind in kinds:
            n = min(_sets(rows, kind), 24) if sweep else _sets(rows, kind)
            xs, rs, sl, sh = [], [], [], []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(1000 * rows + 10 * i + rank)
                rs.append(torch.randn(rows, H, generator=g, device=dev).bfloat16())
                xs.append(torch.randn(rows, H, generator=g, device=dev).bfloat16())
                if kind == "slots":
                    sl.append(torch.randn(rows, 4, H, generator=g, device=dev).bfloat16())
                    sh.append(torch.randn(rows, H, generator=g, device=dev).bfloat16())
            tile = 16 if rows <= 512 else 0

            def norm8(y, r):
                if tile:
                    return an.gemma_add_rmsnorm_mxfp8(y, r, w, EPS, tile)
                return an.gemma_add_rmsnorm_mxfp8_row_major(y, r, w, EPS)

            if kind == "bf16":
                if rows <= 80:
                    def base(i):
                        return ca.custom_fused_ar_rms(xs[i], rs[i], w, EPS, use_1stage=True, gemma_norm=True)
                else:
                    def base(i):
                        return an.gemma_add_rmsnorm(ca.custom_all_reduce(xs[i]), rs[i], w, EPS)

                def entry(i):
                    return staged.ar_add_rmsnorm(xs[i], rs[i], w, EPS)
            elif kind == "mxfp8":
                def base(i):
                    return norm8(ca.custom_all_reduce(xs[i]), rs[i])

                def entry(i):
                    return staged.ar_add_rmsnorm_mxfp8(xs[i], rs[i], w, EPS, tile)
            else:
                def base(i):
                    return norm8(ca.custom_all_reduce(xs[i]), rs[i])

                def entry(i):
                    return slots_comm.slots4_ar_add_rmsnorm_mxfp8(sl[i], sh[i], rs[i], w, EPS, tile)

            arms = {"base": base, "entry": entry}
            if sweep:
                arms = {}
                for cfg in _sweep_configs(rows):
                    def run(i, cfg=cfg):
                        buffers.config = lambda *a, cfg=cfg: cfg
                        try:
                            return entry(i)
                        finally:
                            buffers.config = None
                    arms[cfg] = run
                for fn in arms.values():  # each candidate compiles collectively
                    fn(0)
            base(0)
            entry(0)
            torch.cuda.synchronize()
            graphs = {}
            for name, fn in arms.items():
                gs = []
                for lo in range(0, n, MAX_GRAPH_CALLS):
                    g = torch.cuda.CUDAGraph()
                    with ca.capture(), torch.cuda.graph(g):
                        for i in range(lo, min(n, lo + MAX_GRAPH_CALLS)):
                            fn(i)
                    gs.append(g)
                graphs[name] = gs
            for gs in graphs.values():
                for g in gs:
                    g.replay()
            torch.cuda.synchronize()
            times = {k: [] for k in graphs}
            for rnd in range(rounds):
                order = list(graphs) if rnd % 2 == 0 else list(reversed(list(graphs)))
                for k in order:
                    dist.barrier()
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for _ in range(2):
                        for g in graphs[k]:
                            g.replay()
                    e.record()
                    torch.cuda.synchronize()
                    times[k].append(s.elapsed_time(e) * 1e3 / (2 * n))
            every = [None, None]
            dist.all_gather_object(every, times)
            if rank == 0:
                samples = {k: [max(ev[k][j] for ev in every) for j in range(rounds)] for k in times}
                med = {k: statistics.median(v) for k, v in samples.items()}
                roof = rows * H * 2 / (LINK_GBPS * 1e3)
                if sweep:
                    best = sorted(med.items(), key=lambda kv: kv[1])
                    raw = [kv for kv in best if not kv[0][3]][:2]
                    print(f"rows={rows} {kind} sweep best: " + ", ".join(f"{k}: {v:.2f}" for k, v in best[:4] + raw),
                          flush=True)
                else:
                    parts = [f"rows={rows} {kind} sets={n}"]
                    for k, v in samples.items():
                        parts.append(f"{k} {med[k]:.2f} [{min(v):.2f}-{max(v):.2f}]")
                    parts.append(f"link roofline {roof:.2f} us; entry/base {med['entry'] / med['base']:.3f}")
                    print(" | ".join(parts), flush=True)
                results[f"{rows}_{kind}"] = {"samples": {str(k): v for k, v in samples.items()},
                                             "link_roofline_us": roof}
            del graphs, xs, rs, sl, sh
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    if rank == 0 and out_json:
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=1)
    ca.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rows", nargs="*", type=int)
    p.add_argument("--json")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--kinds", default="bf16,mxfp8,slots")
    p.add_argument("--sweep", action="store_true")
    a = p.parse_args()
    run_on_2_gpus(_worker, a.rows or ROWS, a.kinds.split(","), a.rounds, a.json, a.sweep, timeout=7200)


if __name__ == "__main__":
    main()
