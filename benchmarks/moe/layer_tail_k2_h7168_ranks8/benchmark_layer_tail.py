# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Layer tail graph latency on 8 ranks (torchrun --nproc_per_node 8), with the roofline table.

Each graph replays NSET launches over distinct weights (the working set exceeds the LLC); the time is the median of
ROUNDS x REPL replays, per launch, the slowest rank. ``P`` limits the experts the routing can pick (896: uniform,
the worst case for the expert weights' bytes)."""
import os
import statistics

import torch
import torch.distributed as dist

from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import K2Build, LayerTailRunner, k2_launch
from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import kernel as K
from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import reference as ref
from tests.moe.layer_tail_k2_h7168_ranks8.inputs import layer_inputs, moe_weights

HBM_READ_TBS = 6.4  # measured achievable HBM read on MI355X
NSET, REPL, ROUNDS = 4, 20, 12
bf = torch.bfloat16
DENSE_MB = (7168 * 1536 + 112 * 7168 + 448 * 7168 + 1536 * 7168 + 7168 * 768 + 896 * 3584) * 2 / 1e6
EXPERT_MB = (768 * 1792 + 768 * 112 + 3584 * 192 + 3584 * 12) / 1e6  # w13 + scales, w2 + scales a rank holds


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    device, group = torch.device("cuda", rank), dist.group.WORLD
    comm = LayerTailRunner(device, group, rank)
    sets = []
    for i in range(NSET):
        nat, dev = moe_weights(rank, device)
        sets.append(dev)
    rows = []
    for uniq in (896, 96):
        for S in (6, 12):
            for dev in sets:
                bias = dev["bias"]
                bias.zero_()
                if uniq < 896:
                    bias[torch.randperm(896)[:uniq].to(device)] = 10.0
            ws = [layer_inputs(rank, device, S, seed=i) for i in range(NSET)]
            outs = [torch.empty(S, 7168, dtype=bf, device=device) for _ in range(NSET)]
            prefs = [w["prefix"].clone() for w in ws]

            def step():
                comm.step_begin()
                for i in range(NSET):
                    w, dev = ws[i], sets[i]
                    k2_launch(K2Build(tokens=S, nblocks=4), core=w["core"], w_o=w["w_o"], prefix=prefs[i],
                              blocks=w["blocks"], ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
                              out=outs[i], layer=128 + 3 + i, **{k: dev[k] for k in (
                                  "w_gate", "bias", "w_ld", "w_sgu", "w_sd", "w13", "w13s", "w2", "w2s", "ln_w", "w_up")},
                              **comm.launch_args(), **comm.peer_args())  # fmt: skip

            step()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            graph.replay()
            torch.accelerator.synchronize()
            at = K.scratch_layout(K2Build(tokens=S, nblocks=4))["xrow"][0]
            x = comm.scratch[at : at + S * 7168 * 2].view(bf).view(S, 7168)
            ids, _ = ref.route(x, sets[NSET - 1]["w_gate"], sets[NSET - 1]["bias"])
            used = ids.unique().numel()
            ts = []
            for _ in range(ROUNDS):
                dist.barrier(group)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(REPL):
                    graph.replay()
                b.record()
                torch.accelerator.synchronize()
                ts.append(a.elapsed_time(b) * 1000 / REPL / NSET)
            t = torch.tensor([statistics.median(ts), min(ts), max(ts)], device=device)
            gathered = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(gathered, t)
            if rank == 0:
                m = torch.stack(gathered)
                mb = DENSE_MB + used * EXPERT_MB
                us = m[:, 0].max().item()
                rows.append((S, uniq, used, us, m[:, 1].max().item(), mb, mb / us / HBM_READ_TBS * 100))
    if rank == 0:
        print("M  P   experts  us/layer (min)   MB/rank   TB/s    % of the 6.4 TB/s read ceiling")
        for S, uniq, used, us, mn, mb, pct in rows:
            print(f"{S:<3}{uniq:<5}{used:<8}{us:7.1f} ({mn:6.1f}){mb:9.0f}{mb / us:8.2f}  {pct:5.1f}")


if __name__ == "__main__":
    main()
