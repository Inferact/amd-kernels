# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The phases of SeqParallelHandoffs on 4 ranks, bit for bit against their kernels run on one GPU per shard (every rank
rebuilds every rank's seeded inputs): the input quant + gather, the attention all-to-all + o_proj + norm (+ MoE
preparation) + gather of a MoE and a dense layer, the FFN reduce-scatter + next norm / quant + gather of both, the
final reduce-scatter and the bf16 gather; at the served token counts (tails off the 32-row grid included). The MoE
FFN's norm + quant has no one-GPU kernel: its codes and scales meet ``norm/gemma_add_rmsnorm_h6144/reference.py``'s
tolerances, its residual bit for bit."""

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

from .workload import H, Q, SERVED_TOKENS, WORLD, moe_weights, o_proj_weight, rand

EPS = 1e-6


def _expect(errors, what, got, want):
    if got.shape != want.shape:
        errors.append(f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}")
    elif not torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)):
        bad = (got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).reshape(got.shape[0], -1)
        errors.append(f"{what}: rows {bad.any(dim=1).nonzero().flatten()[:6].tolist()} differ")


def _expect_mxfp8(errors, what, codes, scales, y64):
    """Row-major codes and AITER-shuffled scales of a shard against the fp64 norm (the reference's tolerances)."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import reference as R

    st = R.compare_mxfp8(codes.contiguous().view(torch.uint8), R.unshuffle_scales(scales, codes.shape[0]), y64)
    if (st["scale_max_diff"] > 1 or st["scale_mismatch"] > R.SCALE_MISMATCH or st["code_mismatch"] > R.CODE_MISMATCH
            or st["code_far"] or st["cos"] < R.MXFP8_COS):  # fmt: skip
        errors.append(f"{what}: {st}")


def _unpacked(attn, t, rows, p, dev):
    """Rank p's o_proj input from every rank's attention output: codes [rows, 4 Q], scales in the GEMM's layout."""
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import mxfp8_pack_q2048
    from amd_kernels.gemm.mxfp8_common.gemm import _preshuffle_activation_scale

    codes, scales = [], []
    for a in attn:
        packed = torch.zeros((WORLD * rows, Q + Q // 32), dtype=torch.uint8, device=dev)
        mxfp8_pack_q2048(a, out=packed[:t])
        block = packed[p * rows : (p + 1) * rows]
        codes.append(block[:, :Q])
        scales.append(block[:, Q:])
    return torch.cat(codes, 1).contiguous().view(torch.float8_e4m3fn), _preshuffle_activation_scale(
        torch.cat(scales, 1).contiguous()
    )


class _Delayed:
    """The SDMA phases with a ``between`` that delays this rank's stream by a rank-dependent time, as the other
    microbatch's compute would: the pushes and waits must order the bytes, not the timing."""

    def __init__(self, ph, rank):
        self.ph, self.cycles = ph, 50_000 * (1 + rank)

    def __getattr__(self, name):
        fn = getattr(self.ph, name)
        if name not in ("input_gather", "attention_moe", "attention_dense", "ffn_norm_gather", "ffn_reduce_scatter",
                        "gather"):  # fmt: skip
            return fn
        return lambda *a, **k: fn(*a, between=lambda: torch.cuda._sleep(self.cycles), **k)


def _worker(rank, errors, token_counts, sdma=False):
    from amd_kernels.collectives.pull_collectives_ranks4.reference import (
        reduce_scatter_reference,
        slots4_reduce_scatter_reference,
    )
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import SeqParallelHandoffs, shard_rows
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl, mxfp8_gemm
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import prepare_mxfp4_moe
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference

    dev = torch.device("cuda", rank)
    if sdma:
        from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import SdmaSeqParallelHandoffs

        ph = _Delayed(SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts)), rank)
    else:
        ph = SeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts))
    o_w, o_s = o_proj_weight(dev)
    mw = moe_weights(dev)
    nw = rand((H,), 3, dev, 0.2)

    def o_proj(xq, xs):
        return mxfp8_gemm(xq, xs, o_w, o_s, torch.bfloat16, activation_scale_preshuffled=True)

    for t in token_counts:
        rows = shard_rows(t)
        padded = WORLD * rows
        tag = f"rank {rank} t {t}"
        seed = t * 100
        # 1. the first layer's input
        xs = [rand((rows, H), seed + p, dev) for p in range(WORLD)]
        codes, scales = ph.input_gather(xs[rank])
        want = [mxfp8_e4m3_quantize_flydsl(x) for x in xs]
        torch.cuda.synchronize()
        _expect(errors, f"{tag} input codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} input scales", scales, torch.cat([w[1] for w in want]))
        # 2. the attention output of a MoE layer, then of a dense layer
        attn = [rand((t, Q), seed + 10 + p, dev, 0.5) for p in range(WORLD)]
        res = [rand((rows, H), seed + 20 + p, dev) for p in range(WORLD)]
        moe_in, r = ph.attention_moe(attn[rank], o_w, o_s, res[rank], nw, EPS, mw)
        local, rs = [], []
        for p in range(WORLD):
            y, rp = gemma_add_rmsnorm(o_proj(*_unpacked(attn, t, rows, p, dev)), res[p], nw, EPS)
            local.append(prepare_mxfp4_moe(y, mw).compact())
            rs.append(rp)
        torch.cuda.synchronize()
        for k, field in enumerate(moe_in.tensors()):
            _expect(errors, f"{tag} moe input field {k}", field, torch.cat([c.tensors()[k] for c in local]))
        _expect(errors, f"{tag} moe residual", r, rs[rank])
        codes, scales, r = ph.attention_dense(attn[rank], o_w, o_s, res[rank], nw, EPS)
        want = [gemma_add_rmsnorm_mxfp8_row_major(o_proj(*_unpacked(attn, t, rows, p, dev)), res[p], nw, EPS)
                for p in range(WORLD)]  # fmt: skip
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} dense scales", scales, torch.cat([w[1] for w in want]))
        _expect(errors, f"{tag} dense residual", r, want[rank][2])
        # 3. the FFN output: MoE partials, then a dense partial; the last layer's reduce-scatter; the gather
        slots = [rand((padded, 4, H), seed + 30 + p, dev, 0.25) for p in range(WORLD)]
        shared = [rand((padded, H), seed + 40 + p, dev, 0.25) for p in range(WORLD)]
        codes, scales, r = ph.ffn_norm_gather((slots[rank], shared[rank]), res[rank], nw, EPS)
        sums = [slots4_reduce_scatter_reference(slots, shared, p) for p in range(WORLD)]
        want = [add_rmsnorm_reference(s, res[p], nw, EPS) for p, s in enumerate(sums)]
        torch.cuda.synchronize()
        for p, (y64, _) in enumerate(want):
            blk = slice(p * rows, (p + 1) * rows)
            _expect_mxfp8(errors, f"{tag} slots codes / scales of rank {p}", codes[blk], scales[blk], y64)
        _expect(errors, f"{tag} slots residual", r, want[rank][1])
        h = ph.ffn_reduce_scatter((slots[rank], shared[rank]))
        torch.cuda.synchronize()
        _expect(errors, f"{tag} slots reduce-scatter", h, sums[rank])
        del slots, shared
        dense = [rand((padded, H), seed + 50 + p, dev, 0.25) for p in range(WORLD)]
        codes, scales, r = ph.ffn_norm_gather(dense[rank], res[rank], nw, EPS)
        sums = [reduce_scatter_reference(dense, p) for p in range(WORLD)]
        want = [gemma_add_rmsnorm_mxfp8_row_major(s, res[p], nw, EPS) for p, s in enumerate(sums)]
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense ffn codes", codes, torch.cat([w[0] for w in want]))
        _expect(errors, f"{tag} dense ffn scales", scales, torch.cat([w[1] for w in want]))
        _expect(errors, f"{tag} dense ffn residual", r, want[rank][2])
        h = ph.ffn_reduce_scatter(dense[rank])
        out = ph.gather(res[rank])
        torch.cuda.synchronize()
        _expect(errors, f"{tag} dense reduce-scatter", h, sums[rank])
        _expect(errors, f"{tag} gather", out, torch.cat(res))
        del dense
        torch.cuda.empty_cache()


@pytest.mark.parametrize("tokens", [SERVED_TOKENS])
def test_handoffs_bitwise(tokens):
    run_on_4_gpus(_worker, tokens, timeout=1500)


@pytest.mark.parametrize("tokens", [SERVED_TOKENS])
def test_handoffs_sdma_bitwise(tokens):
    run_on_4_gpus(_worker, tokens, True, timeout=1500)
