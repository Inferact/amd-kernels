# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The op entries' new forms: blockers on the op's arguments decline the rows where the served path wins, and
``prepare_*`` binds a weight or builds state once (a handle) or returns the reason. Host metadata only (meta tensors,
stand-in packages): no GPU."""

from types import SimpleNamespace

import pytest
import torch

from amd_kernels import attention, gemm, indexer, moe, norm, sampling
from amd_kernels.gemm.mxfp8_common import grouped, postmix
from amd_kernels.routes import Prepared

F8, U8, BF = torch.float8_e4m3fn, torch.uint8, torch.bfloat16


def meta(*shape, dtype=BF):
    return torch.empty(shape, dtype=dtype, device="meta")


def spans(blocker, rows):
    return [m for m in rows if blocker(m) is None]


def test_sparse_mla_and_indexer_rows(monkeypatch):
    from amd_kernels.attention.sparse_mla_decode_common import host as sparse_host
    from amd_kernels.indexer.mxfp4_indexer_topk_common import topk as topk_host

    monkeypatch.setattr(sparse_host._ROUTES, "package", lambda key: SimpleNamespace(
        supports_sparse_mla_decode=lambda *a: True))
    monkeypatch.setattr(topk_host._ROUTES, "package", lambda key: SimpleNamespace(blocker=lambda *a: None))
    cache = meta(8, 64, 512, dtype=F8)
    sparse = lambda m: attention.sparse_mla_decode_blocker(meta(m, 64, 512, dtype=F8), cache, *[None] * 13)  # noqa
    assert spans(sparse, (1, 768, 769)) == [1, 768]
    assert attention.sparse_mla_decode_blocker(meta(769, 64, 512, dtype=F8), cache) is None  # previous form
    q = meta(1, 32, 64, dtype=U8)
    topk = lambda m: indexer.mxfp4_indexer_topk_blocker(meta(m, 32, 64, dtype=U8), 1, 2, meta(8, 64, 132, dtype=U8),
                                                        3, 4, 5, next_n=1, max_ctx=4096)  # noqa: E731
    assert spans(topk, (1, 768, 769)) == [1, 768], topk(769)
    h = indexer.prepare_mxfp4_indexer_topk(32, 128, 64, 4096, 1)
    assert isinstance(h, Prepared) and h.blocker(q) is None and "769 rows" in h.blocker(meta(769, 32, 64))
    assert "no kernel" in indexer.prepare_mxfp4_indexer_topk(64, 128, 64, 4096, 1)


def test_mhc_seam_rows_and_aux():
    seam = norm.mhc_post_pre_rmsnorm_blocker
    rows = (1, 288, 289, 512, 513, 720, 721, 4096)
    assert spans(lambda m: seam(meta(m, 4, 5120), 1, 2, 3, 4), rows) == [1, 288, 513, 720]
    assert spans(lambda m: seam(meta(m, 4, 5120), 1, 2, 3, 4, aux_mean=True), rows) == list(rows)
    assert spans(lambda m: seam(meta(m, 4, 5120)), rows) == list(rows)  # previous form: no row ranges


def test_postmix_rows():
    w8, w2 = meta(5120, 8192, dtype=F8), meta(5120, 2304, dtype=F8)
    wo_b = lambda m: gemm.mxfp8_postmix_gemm_blocker(meta(m, 8192, dtype=F8), 1, w8, 2, 3, 4, 5)  # noqa: E731
    ffn = lambda m: gemm.mxfp8_postmix_gemm_blocker(meta(m, 2304, dtype=F8), 1, w2, 2, 3, 4, 5, routed=6)  # noqa
    assert spans(wo_b, (1, 64, 65, 768)) == [1, 64]
    assert spans(ffn, (1, 192, 193, 768)) == [1, 192]
    assert "rows outside" in gemm.mxfp8_postmix_gemm_blocker(meta(769, 2304, dtype=F8), 1, w2, 2, 3, 4, 5, 6)


def test_lm_head_and_markov_rows(monkeypatch):
    from amd_kernels import routes

    monkeypatch.setattr(routes, "available", lambda: True)
    w = meta(129280, 5120)
    lm = lambda m: gemm.bf16_gemm_blocker(meta(m, 5120), w)  # noqa: E731
    assert spans(lm, (112, 113, 128, 129, 495, 496, 512, 513)) == [113, 128, 496, 512]
    assert gemm.bf16_gemm_blocker(meta(16, 6144), meta(50016, 6144)) is None  # no row ranges measured there
    w1 = meta(129280, 256)
    mk = lambda m: sampling.markov_chain_blocker(1, w1, 2, 3, 4, meta(m, 5, dtype=torch.int64))  # noqa: E731
    assert spans(mk, (1, 160, 161)) == [1, 160]


class Grouped(SimpleNamespace):
    MAX_M = 1024

    def grouped_gemm_mxfp8(self, *a):
        return "decode"

    def mxfp8_bmm_mxfp8_out(self, *a):
        return "large"

    def warmup_grouped_gemm(self, device):
        self.warmed = True


def test_grouped_mxfp8_out_takes_large_m_to_the_batched_package(monkeypatch):
    pkg = Grouped()
    monkeypatch.setattr(grouped._ROUTES, "package", lambda key: pkg)
    monkeypatch.setattr(grouped._LARGE_M, "package", lambda key: pkg)
    w, s = meta(8192, 4096, dtype=F8), meta(8, 32, 128, dtype=U8)
    x = lambda m: (meta(m, 8 * 4096, dtype=F8), meta(m, 8 * 128, dtype=U8))  # noqa: E731
    assert gemm.mxfp8_grouped_gemm_mxfp8(*x(1024), w, s) == "decode"
    assert gemm.mxfp8_grouped_gemm_mxfp8(*x(1025), w, s) == "large"
    assert gemm.mxfp8_grouped_gemm_mxfp8_blocker(*x(16384), w, s) is None
    assert "rows outside" in gemm.mxfp8_grouped_gemm_mxfp8_blocker(*x(16385), w, s)
    assert "rows outside" in gemm.mxfp8_grouped_gemm_blocker(*x(1025), w, s)  # bf16 out: decode package only
    h = gemm.prepare_mxfp8_grouped_gemm(w, meta(256, 128, dtype=U8), 8, max_num_tokens=1024)
    assert isinstance(h, Prepared) and pkg.warmed and tuple(h.weight_scale.shape) == (8, 32, 128)
    assert h.try_call(*x(2048)) == "large"
    assert "no kernel" in gemm.prepare_mxfp8_grouped_gemm(meta(4096, 4096, dtype=F8), s, 4)


def test_prepare_mxfp8_linear_binds_the_preshuffled_weight(monkeypatch):
    from amd_kernels.gemm.mxfp8_common import gemm as G, prepared

    seen = {}
    monkeypatch.setattr(G, "warmup_mxfp8_gemm_weight", lambda w, s, n, d: seen.update(warm=(w, n)))
    monkeypatch.setattr(prepared, "preshuffled", lambda w, s, p=None: (w if p is None else p, s))
    w, aiter = meta(1792, 5120, dtype=F8), meta(1792, 5120, dtype=F8)
    h = gemm.prepare_mxfp8_linear(w, meta(56, 160, dtype=U8), weight_preshuffled=aiter)
    assert h.weight is aiter and seen["warm"] == (aiter, 1024)
    assert h.blocker(meta(16, 5120)) is None
    assert h.blocker(meta(16, 5120, dtype=F8), meta(16, 160, dtype=U8)) is not None  # codes: no tuned row
    assert "no shape folder" in gemm.prepare_mxfp8_linear(meta(999, 5120, dtype=F8), meta(999, 160, dtype=U8))
    assert "e4m3" in gemm.prepare_mxfp8_linear(meta(1792, 5120), meta(1792, 160, dtype=U8))


def test_prepare_postmix_and_gate_up(monkeypatch):
    from amd_kernels.gemm.mxfp8_common import gate_up_swiglu, prepared

    monkeypatch.setattr(prepared, "preshuffled", lambda w, s, p=None: (w, s))
    monkeypatch.setattr(postmix, "warmup_mxfp8_postmix_gemm", lambda w, s, r: None)
    monkeypatch.setattr(postmix, "mxfp8_postmix_pack_gate_weight", lambda fn, n, k, r: "packed")
    h = gemm.prepare_mxfp8_postmix_gemm(meta(5120, 8192, dtype=F8), meta(160, 256, dtype=U8), False, fn=meta(24, 20480))
    assert h.fn_packed == "packed" and h.blocker(meta(64, 8192, dtype=F8), 1) is None
    assert h.blocker(meta(65, 8192, dtype=F8), 1) is not None
    assert "no kernel" in gemm.prepare_mxfp8_postmix_gemm(meta(5120, 4096, dtype=F8), meta(5120, 128, dtype=U8), False)
    monkeypatch.setattr(prepared, "prepare_mxfp8_linear",
                        lambda w, s, **kw: SimpleNamespace(weight=w, weight_scale=s))
    g = gate_up_swiglu.prepare_mxfp8_gate_up_swiglu_mxfp8(meta(4608, 5120, dtype=F8), meta(4608, 160, dtype=U8))
    assert isinstance(g, Prepared)
    assert g.blocker(meta(192, 5120)) is None and g.blocker(meta(193, 5120)) is not None


def test_prepare_router_and_expert_parallel(monkeypatch):
    from amd_kernels.moe.mxfp4_experts_common import host as experts
    from amd_kernels.moe.router_topk_common import host as router

    pkg = SimpleNamespace(shuffle_gate=lambda w: "shuffled", warmup_router_topk=lambda d: None,
                          router_topk=lambda *a, **kw: kw)
    monkeypatch.setattr(router._ROUTES, "package", lambda key: pkg)
    h = moe.prepare_router_topk(meta(384, 5120), meta(384, dtype=torch.float32), 6)
    assert h.weight == "shuffled" and h(meta(4, 5120)) == {"weight_shuffled": True}
    assert "no kernel" in moe.prepare_router_topk(meta(256, 7168), meta(256, dtype=torch.float32), 8)
    ep = SimpleNamespace(host=SimpleNamespace(NE_MAX=63, MAX_TOKENS=1024),
                         ExpertParallel=lambda *a, **kw: ("ep", a, kw))
    monkeypatch.setattr(experts._FUSED_EP, "package", lambda key: ep)
    kw = dict(hidden=5120, inter=2304, top_k=6)
    assert moe.prepare_expert_parallel(0, 8, "g", "d", **kw, experts_per_rank=49)[0] == "ep"
    assert "experts per rank" in moe.prepare_expert_parallel(0, 8, "g", "d", **kw, experts_per_rank=64)
    assert "tokens per rank" in moe.prepare_expert_parallel(0, 8, "g", "d", **kw, max_tokens=2048)
    assert moe.expert_parallel_blocker(0, 8, "g", "d", **kw) is None
