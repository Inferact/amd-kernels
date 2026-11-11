# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Non-temporal output stores on the decode rows of ``_NT_OUT_ROWS`` change no byte.

Every listed row is reached through the public entry (``defer_reduce=True``, the TP4 QKV
partial slabs) at the edges of its window; the default-policy kernel (``OUT_CM=0``, the
name without ``_ont``) runs the same call through a swapped launcher and gives the same
slabs, deterministically, also after a poisoned CUDA graph replay. Rows outside the table
keep their kernel name (the default stores)."""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import mxfp8_gemm_decode as d
from .test_gemm import make_inputs, require_gfx950

N, K = 2560, 6144
EDGES = (81, 96, 97, 113, 128, 129, 144, 145, 192, 193, 256, 257, 288)


def _launcher(c, out_cm):
    return d.compile_mxfp8_gemm_decode(
        N=N, K=K, RT=max(1, c.tile_m // 16), NI=c.tile_n // 16, KW=c.waves_m, KS=c.split_k,
        LA=(c.la_w, c.la_a) if c.la_w else None, sched_barriers=bool(c.waves_n), W_CM=c.w_cm,
        OUT_CM=out_cm,
    )


def _config(m):
    return g._deferred_qkv_config(m, N, K, g._select_prequant_config(m, N, K, torch.bfloat16))


@pytest.mark.parametrize('m', EDGES)
@torch.inference_mode()
def test_nt_out_rows_bitwise(m, monkeypatch):
    require_gfx950()
    x, _, _, (w, s) = make_inputs(m, N, K, seed=m + 7)
    x[2].zero_()
    c = _config(m)
    key = (N, K, max(1, c.tile_m // 16), c.tile_n // 16, c.waves_m, c.split_k)
    assert c.kernel == g._DECODE_KERNEL and key in d._NT_OUT_ROWS, (m, c)
    nt = g._decode_launcher(N, K, c, False)
    plain = _launcher(c, 0)
    assert nt.kernel_name == plain.kernel_name + '_ont', (nt.kernel_name, plain.kernel_name)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, g.prequant_code_tile_rows(m, N, K))

    def run():
        y = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m, defer_reduce=True)
        assert isinstance(y, g.SkPartials) and y.split_k == c.split_k
        return y.tmp

    got = run().clone()
    assert torch.equal(got, run())  # deterministic
    with monkeypatch.context() as mp:
        mp.setattr(g, '_decode_launcher', lambda n_, k_, c_, a_bf16: plain)
        assert torch.equal(got, run())
    assert got[:, 2].count_nonzero() == 0
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(3):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)


@pytest.mark.parametrize('m', (16, 64, 80, 289, 512))
def test_other_rows_keep_default_stores(m):
    c = _config(m)
    if c.kernel != g._DECODE_KERNEL:
        pytest.skip(c.kernel)
    key = (N, K, max(1, c.tile_m // 16), c.tile_n // 16, c.waves_m, c.split_k)
    assert key not in d._NT_OUT_ROWS
    assert not g._decode_launcher(N, K, c, False).kernel_name.endswith('_ont')
