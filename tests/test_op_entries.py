# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The shape-agnostic op entries (``amd_kernels.routes``, one per op contract): each routes a call by the shape key it
reads from its tensors to the registered shape package, and declines an uncovered call with a reason. Host metadata
only (meta tensors, a stand-in package): no GPU."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

import amd_kernels.activation as activation
import amd_kernels.attention as attention
import amd_kernels.engram as engram
import amd_kernels.gemm as gemm
import amd_kernels.indexer as indexer
import amd_kernels.moe as moe
import amd_kernels.norm as norm
import amd_kernels.sampling as sampling
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.routes import Routes

ROUTES = {
    f"{mod.__name__}.{name}": routes
    for mod in (sys.modules[m] for m in sorted(sys.modules) if m.startswith("amd_kernels.") and "_common." in m)
    for name, routes in vars(mod).items()
    if isinstance(routes, Routes)
}


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


class Stub(SimpleNamespace):
    """A stand-in shape package: every function records its call and accepts the inputs."""

    def __getattr__(self, name):
        if name.startswith("supports") or name in ("serves",):
            return lambda *a, **k: True
        if name == "blocker":
            return lambda *a, **k: None

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return name

        return call


@pytest.fixture
def stub(monkeypatch):
    """Every registry's packages replaced by one ``Stub`` (keys unchanged)."""
    s = Stub(calls=[], MAX_ROWS=768, MAX_M=1024, host=SimpleNamespace(NE_MAX=63, MAX_TOKENS=1024),
             kernel=SimpleNamespace(MAX_REQS=1024, SWIGLU=(), SWIGLU_PREQUANT=()))
    for routes in ROUTES.values():
        monkeypatch.setattr(routes, "package", lambda key, _r=routes: s if key in _r.hosts else Routes.package(_r, key))
    return s


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_every_route_names_a_package(name):
    routes = ROUTES[name]
    assert routes.shapes()
    for key, module in routes.hosts.items():
        assert importlib.util.find_spec(module) is not None, (name, key, module)
        assert routes.blocker(key) is None


def test_uncovered_keys_decline_with_a_reason():
    cases = [
        attention.sparse_mla_decode_blocker(meta(4, 128, 512), meta(8, 64, 512)),
        attention.rope_fp8_kv_insert_blocker(meta(4, 64, 576), meta(8, 64, 576), meta(16, 64, dtype=torch.float32)),
        attention.compressor_insert_blocker(meta(4, 1024, dtype=torch.float32), 2, meta(8, 64, 448), meta(8, 4352),
                                            meta(128, 448), meta(16, 64), meta(16, 64)),
        indexer.mxfp4_indexer_blocker(64, 128, 64, 4096, 1),
        indexer.mxfp4_indexer_blocker(32, 128, 64, indexer.MXFP4_INDEXER_MAX_ENTRIES + 1, 1),
        indexer.q_rope_mxfp4_quant_blocker(meta(4, 64, 128), meta(16, 64)),
        moe.router_topk_blocker(meta(4, 7168), meta(256, dtype=torch.float32), 8),
        moe.mxfp4_experts_blocker(7168, 2048, 32, 8, 16),
        moe.expert_parallel_blocker(4, 5120, 2304, 6, 96, 16),
        norm.rmsnorm_mxfp8_blocker(meta(4, 7168), meta(7168)),
        norm.rmsnorm_dual_mxfp8_blocker(meta(4, 1536), meta(4, 512)),
        norm.mhc_post_pre_rmsnorm_blocker(meta(4, 4, 7168)),
        norm.mhc_post_collapse_rmsnorm_blocker(meta(4, 2, 5120)),
        norm.mhc_gate_finish_blocker(meta(4, 64, 32, dtype=torch.float32), meta(4, 7168)),
        activation.clamp_silu_mul_mxfp8_blocker(meta(4, 2 * 1536)),
        sampling.markov_chain_blocker(meta(200064, 256), meta(4, 5, dtype=torch.int64)),
        sampling.rejection_sample_blocker(meta(4, 3, 129280), 4),
        engram.ngram_rows_blocker(4, 2, 24),
        engram.gated_post_blocker(meta(4, 4, 7168), meta(4, 5 * 7168)),
        gemm.mxfp8_gate_up_swiglu_mxfp8_blocker(16, 6144, 6144),
        gemm.mxfp8_postmix_blocker(16, 5120, 8192, routed=True),
        gemm.mxfp8_grouped_gemm_blocker(16, 4, 1024, 4096),
        gemm.bf16_gemm_blocker(meta(100032, 6144)),
        gemm.bf16_gemm_f32_bf16_blocker(meta(4, 7168), meta(1056, 7168), 1024),
        G.mxfp8_linear_blocker(16, 999, 5120),
    ]
    for i, reason in enumerate(cases):
        assert isinstance(reason, str) and reason, i


def test_limits_decline(stub):
    assert "rows" in gemm.mxfp8_postmix_blocker(769, 5120, 8192, routed=False)
    assert "rows" in gemm.mxfp8_grouped_gemm_blocker(1025, 8, 1024, 4096)
    assert "tokens" in moe.mxfp4_experts_blocker(5120, 2304, 48, 6, 1025)
    assert "experts" in moe.expert_parallel_blocker(8, 5120, 2304, 6, 64, 16)
    assert "rows" in norm.mhc_post_pre_rmsnorm_blocker(meta(norm.MHC_MAX_ROWS + 1, 4, 5120))
    assert "requests" in sampling.rejection_sample_blocker(meta(1025, 5, 129280), 1025)
    assert moe.expert_parallel_blocker(8, 5120, 2304, 6, 48, 1024) is None
    assert gemm.mxfp8_postmix_blocker(768, 5120, 2304, routed=True) is None


def test_entries_route_to_the_shape_package(stub):
    f32, i64 = torch.float32, torch.int64
    q = meta(16, 64, 512, dtype=torch.float8_e4m3fn)
    attention.sparse_mla_decode(q, meta(8, 64, 512, dtype=torch.float8_e4m3fn), *range(13), extra=3)
    attention.rope_fp8_kv_insert(meta(16, 64, 512), 1, 2, meta(8, 64, 512), 3, 4, meta(32, 64, dtype=f32), 5, 6)
    cs = meta(32, 64, dtype=f32)
    attention.compressor_insert(meta(16, 1024, dtype=f32), 1, 2, 3, 4, 5, 6, 1e-6, meta(8, 64, 512), 7, cs, 8,
                                meta(128, 512), 9, 1e-6, meta(8, 4352, dtype=torch.uint8), 10, cs, 2)
    indexer.mxfp4_indexer_topk(meta(16, 32, 64, dtype=torch.uint8), 1, 2, 3, 4, 5, 6, next_n=1, max_ctx=4096)
    indexer.q_rope_mxfp4_quant(meta(16, 32, 128), 1, meta(32, 64, dtype=f32), 2, 1.0)
    moe.router_topk(meta(16, 5120), 1, meta(384, dtype=f32), 6)
    moe.router_topk(meta(16, 5120), 1, meta(128, dtype=f32), 3, weight_shuffled=True)
    norm.rmsnorm_mxfp8(meta(16, 5120), meta(5120), 1e-6)
    norm.rmsnorm_dual_mxfp8(meta(16, 1280), meta(16, 512), 1, 2, 1e-6)
    norm.mhc_post_pre_rmsnorm(meta(16, 4, 5120), 1, 2, 3, 4, quant=True)
    norm.mhc_post_collapse_rmsnorm(meta(16, 4, 5120), 1, 2, eps=1e-6)
    norm.mhc_gate_finish(meta(16, 64, 32, dtype=f32), meta(16, 5120), 1, 2, 3)
    activation.clamp_silu_mul_mxfp8(meta(16, 4608))
    sampling.markov_chain(1, meta(129280, 256), 2, 3, 4, meta(16, 5, dtype=i64), 5, 6, 7, 8, 9)
    sampling.rejection_sample(1, meta(16, 5, 129280), 2, 3, meta(16, dtype=torch.int32), 4, 5)
    engram.gated_post(meta(16, 4, 5120), meta(16, 5 * 5120), 1, 2)
    codes = meta(16, 8192, dtype=torch.float8_e4m3fn)
    gemm.mxfp8_postmix_gemm(codes, 1, meta(5120, 8192), 2, 3, 4, 5)
    gemm.mxfp8_postmix_gemm(meta(16, 2304), 1, meta(5120, 2304), 2, 3, 4, 5, routed=6)
    gemm.mxfp8_grouped_gemm_mxfp8(meta(16, 8 * 4096), 1, meta(8192, 4096), meta(8, 32, 128))
    gemm.bf16_gemm(meta(16, 5120), meta(129280, 5120))
    gemm.bf16_gemm_f32_bf16(meta(16, 5120), meta(1056, 5120), 1, meta(16, 32, dtype=torch.bfloat16))
    called = [name for name, _, _ in stub.calls]
    assert called == [
        "sparse_mla_decode", "rope_fp8_kv_insert", "compressor_insert", "indexer_topk", "q_rope_mxfp4_quant",
        "router_topk", "router_topk", "rmsnorm_mxfp8", "rmsnorm_dual_mxfp8", "mhc_post_pre_rmsnorm",
        "mhc_post_collapse_rmsnorm", "mhc_gate_finish", "clamp_silu_mul_mxfp8", "markov_chain", "rejection_sample",
        "gated_post", "postmix_gemm", "postmix_gemm", "grouped_gemm_mxfp8", "bf16_gemm_n129280_k5120", "gemm",
    ]  # fmt: skip
    # the routed add goes in the package's argument order (residual, routed, post_mix, comb_mix)
    assert stub.calls[17][1][4:8] == (3, 6, 4, 5)
    assert stub.calls[6][2] == {"weight_shuffled": True}


def test_uncovered_call_raises_the_reason():
    with pytest.raises(ValueError, match="no kernel for"):
        norm.rmsnorm_mxfp8(meta(4, 7168), meta(7168), 1e-6)


def test_mxfp8_linear_registry_covers_the_new_shapes():
    for shape in ((1792, 5120), (32768, 1280), (4096, 1280), (4608, 5120), (5120, 2304), (5120, 8192),
                  (5120, 15360), (25600, 6144)):  # fmt: skip
        assert shape in G._ROUTES.hosts and shape in G.tuned_shapes(torch.bfloat16), shape
    assert G.mxfp8_linear_blocker(16, 1792, 5120) is None
    assert G.mxfp8_linear_blocker(32, 4096, 1280, prequant=True) is None
    assert G.mxfp8_linear_blocker(16, 4096, 1280, prequant=True) is not None
    assert G.mxfp8_linear_blocker(16, 5120, 2304, prequant=True) is None
    assert G.mxfp8_linear_blocker(848, 32768, 1280, prequant=True) is None
    assert G.mxfp8_linear_blocker(768, 32768, 1280, prequant=True) is not None
    assert G.mxfp8_linear_blocker(1025, 32768, 1280, prequant=True) is not None
    assert G.mxfp8_linear_blocker(1024, 5120, 8192) is None
    assert G.mxfp8_linear_blocker(16, 5120, 15360) is not None and G.mxfp8_linear_blocker(640, 5120, 15360) is None
    assert G.mxfp8_linear_blocker(16, 1792, 5120, torch.float16) is not None
    assert G.mxfp8_linear_blocker(16, 2560, 6144) is None
