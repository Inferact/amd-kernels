# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Test inputs: requests over an NVFP4 page cache, and their top-k block picks.

K / V Gaussian of std 2 in every slot of every page, encoded as the insert does (``reference.nvfp4_quantize``); the
requests' pages disjoint and shuffled; request r's last ``q_lens[r]`` of ``ctx_lens[r]`` tokens are the queries.
Top-k picks are "random" (own block + 15 random earlier ones, the union's worst case) or "shared" (the indexer's
pattern: a per-request base of 15 earlier blocks, each slot replaced by a random earlier block with probability 0.1).
"""

from dataclasses import dataclass

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import reference as ref

BLOCK, HEAD_DIM, GROUP, TOPK = ref.BLK, ref.D, ref.HG, ref.TOPK
SM_SCALE = HEAD_DIM**-0.5
# served KV heads per GPU: TP4 (16 query heads)
TP4_HKV = 1


def require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx950"):
        pytest.skip(f"the kernels use gfx950 instructions (device: {arch})")


@dataclass
class Case:
    q: torch.Tensor  # [T, 16 Hkv, 128] bf16
    cache: torch.Tensor  # [pages, 2 Hkv, 128, 72] uint8
    topk: torch.Tensor  # [Hkv, T, 16] int32
    table: torch.Tensor  # [reqs, max_blocks] int32
    req: torch.Tensor  # [T] int32
    pos: torch.Tensor  # [T] int32
    hkv: int
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    q_lens: list
    ctx_lens: list

    def args(self, q=None):
        """Positional arguments of ``nvfp4_sparse_prefill`` up to ``sm_scale``."""
        return (
            self.q if q is None else q,
            self.cache,
            self.topk,
            self.table,
            self.req,
            self.pos,
            self.hkv,
            SM_SCALE,
        )

    def rows(self, n=48):
        """Sampled rows: every request's first / last 4 tokens and tile edges, plus
        ``n`` spread over the batch."""
        t = self.q.shape[0]
        picks, start = set(), 0
        for ql in self.q_lens:
            picks.update(range(start, min(start + 4, t)))
            picks.update(range(max(start, start + ql - 4), start + ql))
            start += ql
        picks.update(torch.linspace(0, t - 1, min(n, t)).long().tolist())
        return sorted(picks)


def random_picks(pos, hkv, gen):
    """Own block first, then up to 15 distinct random earlier blocks, -1 padded."""
    n = pos.numel()
    cur = (pos // BLOCK).long()
    width = max(int(cur.max()) + 1, TOPK)
    keys = torch.rand(hkv, n, width, generator=gen)
    j = torch.arange(width)
    keys = keys.masked_fill(j >= cur[:, None], 2.0)  # only earlier blocks
    order = keys.argsort(-1)[..., : TOPK - 1]
    ok = order < cur[None, :, None]
    picks = torch.where(ok, order, torch.full_like(order, -1))
    return torch.cat([cur[None, :, None].expand(hkv, n, 1), picks], -1).int()


def shared_picks(pos, req, hkv, gen):
    """Neighbouring tokens pick mostly the same blocks (module docstring)."""
    n = pos.numel()
    cur = (pos // BLOCK).long()
    first = {}
    for p, r in zip(pos.tolist(), req.tolist()):
        first.setdefault(r, p // BLOCK)
    n_req = int(req.max()) + 1
    base = torch.zeros(n_req, TOPK - 1, dtype=torch.long)
    for r, b in first.items():
        base[r] = torch.randperm(max(b, TOPK), generator=gen)[: TOPK - 1]
    cand = base[req.long()][None].expand(hkv, n, TOPK - 1).clone()
    swap = torch.rand(hkv, n, TOPK - 1, generator=gen) < 0.1
    alt = (torch.rand(hkv, n, TOPK - 1, generator=gen) * cur.clamp(min=1)[:, None]).long()
    cand = torch.where(swap, alt, cand)
    keep = cand < cur[None, :, None]
    # drop repeats of an earlier kept slot (the own block is never < cur)
    same = cand[..., :, None] == cand[..., None, :]
    earlier = torch.ones(TOPK - 1, TOPK - 1, dtype=torch.bool).tril(-1)
    dup = (same & earlier & keep[..., None, :]).any(-1)
    keep &= ~dup
    order = torch.sort((~keep).int(), dim=-1, stable=True).indices  # kept first, in order
    cand = cand.gather(-1, order)
    kept = keep.gather(-1, order)
    picks = torch.where(kept, cand, torch.full_like(cand, -1))
    return torch.cat([cur[None, :, None].expand(hkv, n, 1), picks], -1).int()


def make_case(q_lens, ctx_lens, hkv, scales=(1.0, 1.0), seed=0, picks="random"):
    gen = torch.Generator().manual_seed(seed)
    k_scale, v_scale = (torch.tensor(s, device="cuda") for s in scales)
    nbs = [(c + BLOCK - 1) // BLOCK for c in ctx_lens]
    n_pages = sum(nbs) + 2
    kv = []
    for scale in (k_scale, v_scale):
        x = (torch.randn(n_pages, BLOCK, hkv, HEAD_DIM, generator=gen) * 2).bfloat16()
        x = x.cuda().float().permute(0, 2, 1, 3).contiguous()  # [pages, hkv, tok, dim]
        kv += list(ref.nvfp4_quantize(x, scale))
    cache = ref.pack_kv_pages(*kv)
    perm = torch.randperm(n_pages, generator=gen).int()
    table = torch.zeros(len(q_lens), max(nbs), dtype=torch.int32)
    off = 0
    for r, nb in enumerate(nbs):
        table[r, :nb] = perm[off : off + nb]
        off += nb
    pos = torch.cat([torch.arange(c - q, c) for q, c in zip(q_lens, ctx_lens)]).int()
    req = torch.cat([torch.full((q,), r) for r, q in enumerate(q_lens)]).int()
    if picks == "random":
        topk = random_picks(pos, hkv, gen)
    else:
        topk = shared_picks(pos, req, hkv, gen)
    q = (torch.randn(pos.numel(), hkv * GROUP, HEAD_DIM, generator=gen) * 2).bfloat16()
    return Case(
        q.cuda(),
        cache,
        topk.contiguous().cuda(),
        table.cuda(),
        req.cuda(),
        pos.cuda(),
        hkv,
        k_scale,
        v_scale,
        list(q_lens),
        list(ctx_lens),
    )
