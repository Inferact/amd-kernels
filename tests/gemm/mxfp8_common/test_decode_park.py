# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The park-all K-reduce of the decode kernel (``_PARK_ALL_ROWS``) gives the owner form's bits.

Every measured row is reached through the public entry at the edges of its window; the
owner form (``PARK_ALL=False``, kernel name ``..._rk0``) runs the same call through a swapped
launcher. BF16 rows also pass the FP64 reference and replay bit for bit in a CUDA graph."""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_common import mxfp8_gemm_decode as d
from .test_gemm import make_inputs, require_gfx950

ROWS = (
    [(2560, 6144, m, False) for m in (97, 128, 129, 192, 193, 288, 641, 768, 769, 1024, 1025, 1153, 1280)]
    + [(2560, 6144, m, True) for m in (129, 144, 145, 192, 193, 256, 257, 288)]
    + [(6144, 2048, m, False) for m in (129, 145, 160, 161, 192, 257, 288, 320)]
)


def _owner_launcher(n, k, c):
    return d.compile_mxfp8_gemm_decode(
        N=n, K=k, RT=max(1, c.tile_m // 16), NI=c.tile_n // 16, KW=c.waves_m, KS=c.split_k,
        LA=(c.la_w, c.la_a) if c.la_w else None, sched_barriers=bool(c.waves_n), W_CM=c.w_cm,
        PARK_ALL=False,
    )


@pytest.mark.parametrize('n,k,m,defer', ROWS)
@torch.inference_mode()
def test_park_all_rows(n, k, m, defer, monkeypatch):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, n, k, seed=m + n)
    x[3].zero_()
    c = g._select_prequant_config(m, n, k, torch.bfloat16)
    if defer:
        c = g._deferred_qkv_config(m, n, k, c)
    rt, ni, kw = max(1, c.tile_m // 16), c.tile_n // 16, c.waves_m
    assert c.kernel == g._DECODE_KERNEL and (n, k, rt, ni, kw, c.split_k) in d._PARK_ALL_ROWS, c
    park = g._decode_launcher(n, k, c, False)
    owner = _owner_launcher(n, k, c)
    assert owner.kernel_name == park.kernel_name + '_rk0', (owner.kernel_name, park.kernel_name)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, c.tile_m)

    def run():
        y = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m, defer_reduce=defer)
        if defer:
            assert isinstance(y, g.SkPartials) and y.split_k == c.split_k
            return y.tmp.clone()
        return y.clone()

    got = run()
    assert torch.equal(got, run())  # deterministic
    with monkeypatch.context() as mp:
        mp.setattr(g, '_decode_launcher', lambda n_, k_, c_, a_bf16: owner)
        assert torch.equal(got, run())
    if defer:
        return
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    mt = ref.compare(got, expected)
    assert mt['cos'] >= ref.COS_MIN and mt['rel_l2'] <= ref.REL_L2_MAX
    assert got[3].count_nonzero() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    for _ in range(3):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)
