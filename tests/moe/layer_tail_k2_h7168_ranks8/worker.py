# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rank body of test_layer_tail (torchrun, 8 ranks): the entry against ``reference`` at the served widths, the
ranks' agreement and a replayed CUDA graph."""
import os
import sys

import torch
import torch.distributed as dist

from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import LayerTailRunner, LayerTailWeights, layer_tail
from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import kernel as K
from amd_kernels.moe.layer_tail_k2_h7168_ranks8.kernel import K2Build
from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import reference as ref

from .inputs import layer_inputs, moe_weights

bf = torch.bfloat16
HIDDEN = ref.HIDDEN


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    comm = LayerTailRunner(device, group, rank)
    nat, dev = moe_weights(rank, device)
    ar = ref.gather_all_reduce(group)
    cases = [(12, False), (6, True), (16, False), (8, True)]
    errors = []
    for S, reset in cases:
        w = layer_inputs(rank, device, S)
        weights = LayerTailWeights(w["w_o"], w["ares_nw"], w["ares_qk"], w["in_nw"], **dev)
        prefix = w["prefix"].clone()
        comm.step_begin()
        out = layer_tail(comm, weights, w["core"], prefix, w["blocks"], nblocks=4, reset=reset, layer=3)
        torch.accelerator.synchronize()
        attn = ar(ref.bf16(w["core"].float() @ w["w_o"].float().t()).to(bf))
        pre, x_ref = ref.attn_res(attn if reset else w["prefix"], None if reset else attn, w["blocks"],
                                  w["ares_nw"], w["ares_qk"], w["in_nw"], 4 + int(reset), 1e-5, 1e-5)
        key = K2Build(tokens=S, nblocks=4 + int(reset), reset=reset)
        at = K.scratch_layout(key)["xrow"][0]
        x = comm.scratch[at : at + S * HIDDEN * 2].view(bf).view(S, HIDDEN)
        got_pre = prefix
        if cos(x, x_ref) < 0.9999 or cos(got_pre, pre) < 0.9999:
            errors.append(f"rank {rank} S={S} reset={reset}: AttnRes cos {cos(x, x_ref):.6f} / {cos(got_pre, pre):.6f}")
        want = ref.moe(x, nat["w_gate"], nat["bias"], nat["w_ld"], nat["w_sgu"], nat["w_sd"], nat["w13"],
                       nat["w13s"], nat["w2"], nat["w2s"], nat["ln_w"], nat["w_up"], rank, ar)
        if cos(out, want) < 0.9999:
            errors.append(f"rank {rank} S={S} reset={reset}: MoE cos {cos(out, want):.6f}")
        outs = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(outs, out)
        if not all(torch.equal(outs[0], o) for o in outs):
            errors.append(f"rank {rank} S={S}: ranks disagree")
    # a graph of one step, replayed
    S = 12
    w = layer_inputs(rank, device, S, seed=1)
    weights = LayerTailWeights(w["w_o"], w["ares_nw"], w["ares_qk"], w["in_nw"], **dev)
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        comm.step_begin()
        layer_tail(comm, weights, w["core"], prefix, w["blocks"], nblocks=4, reset=True, layer=5, out=out)
    graph.replay()
    torch.accelerator.synchronize()
    want_pre, want_out = prefix.clone(), out.clone()
    for _ in range(100):
        graph.replay()
    torch.accelerator.synchronize()
    if not (torch.equal(prefix, want_pre) and torch.equal(out, want_out)):
        errors.append(f"rank {rank}: graph replay differs")
    if comm.health.any() or comm.flags.any():
        errors.append(f"rank {rank}: a wait gave up")
    gathered = [None] * world
    dist.all_gather_object(gathered, errors)
    if rank == 0:
        flat = [e for g in gathered for e in g]
        print("ERRORS:" + repr(flat) if flat else "OK", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
