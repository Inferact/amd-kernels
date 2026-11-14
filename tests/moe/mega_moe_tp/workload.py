# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MegaMoE TP inputs for the tests and the benchmark: the served shapes, random MXFP4 weights of this rank's inter
slice (kernel layout + row-major for ``reference.py``), routed inputs every rank rebuilds from seeds (no tensor
collective: the test ranks run a gloo group), and the reference output summed over the ranks on the host."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from amd_kernels.moe.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig, shuffle_scale, shuffle_weight
from amd_kernels.moe.mega_moe_tp.reference import mega_moe_tp_ref, mxfp4_quant

SEED = 123


@dataclass(frozen=True)
class Shape:
    name: str
    model_dim: int
    inter_dim: int  # all ranks'
    experts: int
    topk: int
    act: str


# m3: H 6144, I 3072 (768 per rank on 4), 128 routed + 1 shared expert (top 4 + shared), SwiGLU-OAI
SHAPES = {
    "m3": Shape("m3", 6144, 3072, 129, 5, "swiglu"),
    "glm5": Shape("glm5", 6144, 2048, 257, 9, "silu"),
}


@dataclass
class Weights:
    inter: int
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w1_ref: torch.Tensor
    w1_scale_ref: torch.Tensor
    w2_ref: torch.Tensor
    w2_scale_ref: torch.Tensor


def _quant_experts(experts, rows, cols, magnitude, seed, device, chunk_bytes=512 << 20):
    chunk = max(1, min(experts, chunk_bytes // max(rows * cols * 2, 1)))
    codes = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((experts, rows, cols // 32), dtype=torch.uint8, device=device)
    gen = torch.Generator(device=device)
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
        gen.manual_seed(seed + start)
        w = torch.randn((n, rows, cols), dtype=torch.bfloat16, device=device, generator=gen)
        codes[start : start + n], scales[start : start + n] = mxfp4_quant(w.mul_(magnitude))
    return codes, scales


def build_weights(shape: Shape, rank: int, world: int, device, seed=SEED) -> Weights:
    inter = shape.inter_dim // world
    H, E = shape.model_dim, shape.experts
    base = seed + 1_000_000 * rank
    w1, s1 = _quant_experts(E, 2 * inter, H, H**-0.25, base, device)
    w2, s2 = _quant_experts(E, H, inter, inter**-0.25, base + 7, device)
    return Weights(inter, shuffle_weight(w1), shuffle_scale(s1.view(E * 2 * inter, -1)), shuffle_weight(w2),
                   shuffle_scale(s2.view(E * H, -1)), w1, s1, w2, s2)


def route(rows, shape: Shape, kind: str, gen: torch.Generator, device):
    """(topk weights fp32, topk ids int32) of ``rows`` tokens: balanced, random, hot (every token on the last topk
    experts) or subset (~55% of the experts)."""
    E, K = shape.experts, shape.topk
    if kind == "balanced":
        start = int(torch.randint(0, E, (1,), device=device, generator=gen))
        ids = (start + torch.arange(rows * K, device=device)) % E
        score = torch.full((rows, E), -1e4, device=device)
        score.scatter_(1, ids.view(rows, K), 0.0)
        score += 1e-3 * torch.randn((rows, E), device=device, generator=gen)
    else:
        score = torch.randn((rows, E), device=device, generator=gen)
        if kind == "hot":
            score[:, E - K :] += 100.0
        elif kind == "subset":
            score[:, E * 35 // 64 :] = -1e4
    w, ids = torch.softmax(score, dim=-1).topk(K, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.float().contiguous(), ids.to(torch.int32).contiguous()


def _inputs(shape, rows, kind, seed, device, mask):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((rows, shape.model_dim), dtype=torch.bfloat16, device=device, generator=gen)
    w, ids = route(rows, shape, kind, gen, device)
    if mask:
        # a fraction of the routes masked: id -1 or an id >= E (no expert)
        sel = torch.rand(ids.shape, device=device, generator=gen)
        ids = torch.where(sel < mask / 2, -1, torch.where(sel < mask, shape.experts + 3, ids)).to(torch.int32)
    return x, w, ids.contiguous()


@dataclass
class Case:
    x: torch.Tensor  # this rank's input (ag_rs: its m rows; ar: all M rows)
    w: torch.Tensor
    ids: torch.Tensor
    ref: torch.Tensor  # fp32, the rows this rank's output holds
    ids_all: torch.Tensor  # every token's ids


def make_case(shape: Shape, wt: Weights, rank: int, world: int, mode: str, tokens: int, kind: str, seed: int,
              mask: float = 0.0) -> Case:
    """``tokens`` global tokens; ag_rs: rank r's rows from seed + 7919 r, ar: one replicated batch from seed."""
    dev = wt.w1.device
    if mode == "ag_rs":
        m = tokens // world
        parts = [_inputs(shape, m, kind, seed + 7919 * r, dev, mask) for r in range(world)]
        x, w, ids = (torch.cat(t) for t in zip(*parts))
        own = parts[rank]
    else:
        x, w, ids = own = _inputs(shape, tokens, kind, seed, dev, mask)
    part = mega_moe_tp_ref(x, w, ids, wt.w1_ref, wt.w1_scale_ref, wt.w2_ref, wt.w2_scale_ref, shape.act).cpu()
    dist.all_reduce(part)
    ref = part.to(dev)
    if mode == "ag_rs":
        ref = ref[rank * m : (rank + 1) * m]
    return Case(*own, ref, ids)


def new_layer(shape: Shape, wt: Weights, rank: int, world: int, mode: str, max_local_tokens: int,
              comm_dtype: str = "fp8", ar_gather: str = "bf16", **kw) -> MegaMoeTP:
    cfg = MegaMoeTPConfig(rank=rank, world_size=world, model_dim=shape.model_dim, inter_dim=wt.inter,
                          experts=shape.experts, topk=shape.topk, max_local_tokens=max_local_tokens,
                          activation=shape.act, comm_mode=mode, comm_dtype=comm_dtype, ar_gather=ar_gather, **kw)
    return MegaMoeTP(cfg, w1=wt.w1, w1_scale=wt.w1_scale, w2=wt.w2, w2_scale=wt.w2_scale, device=wt.w1.device)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 over every rank's rows (host all-reduce)."""
    d = actual.float() - expected.float()
    t = torch.stack([(d * d).sum(), (expected.float() ** 2).sum()]).cpu()
    dist.all_reduce(t)
    return float((t[0] / t[1]).sqrt()) if t[1] > 0 else float("nan")


def rank_max(v: float) -> float:
    t = torch.tensor([float(v)])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def same_on_all_ranks(t: torch.Tensor) -> bool:
    ref = t.cpu()
    dist.broadcast(ref, src=0)
    return rank_max(0.0 if torch.equal(ref, t.cpu()) else 1.0) == 0.0
