# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Early previous-tile stores preserve FP32 SwiGLU and every valid MXFP8 byte."""
import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import shared_dense as sd
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_gemm1_a8w8_epilogue_early import (
    compile_shared_gemm1_a8w8_epilogue_early,
)
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.mxfp8 import shared_dense_4wave as impl
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
from .test_shared_expert import weights
from .workload import make_activation


def legacy_gemm1():
    return impl.compile_mxfp8_gemm_4w_gateup(
        K=6144, alpha=1.702, beta=1.0, limit=7.0,
        epilogue_type=sd.StoreSharedFp32,
        kernel_prefix='shared_fp32_swiglu_mxfp8_h6144_i768_', tiles=(4, 4))


def row_scales(s, m):
    return s.view(-1, 3, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).reshape(-1, 24)[:m]


@pytest.mark.parametrize('m', [1, 257, 1025, 2048, 4097, 8192, 8193, 10240, 10752, 10753, 16384, 32768, 32769, 87381])
@pytest.mark.parametrize('schedule', ['early', 'single'])
def test_early_epilogue_changed_graph_inputs(weights, m, schedule):
    _, w = weights
    if schedule == 'single' and m > 10752:
        pytest.skip('single-tile grid requires ceil(M/256)*6 <= 256')
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_gemm1_a8w8_single_tile import compile_shared_gemm1_a8w8_single_tile
    old = legacy_gemm1()
    new = (compile_shared_gemm1_a8w8_epilogue_early() if schedule == 'early'
           else compile_shared_gemm1_a8w8_single_tile())
    x = make_activation(m, seed=m)
    q, s = shared_quant(x)
    padded = (m + 31) // 32 * 32
    ss = torch.empty((padded, 192), device='cuda', dtype=torch.uint8)
    _run_compiled(sd._shuffle_scales(6144), s.view(-1), ss.view(-1), m, torch.cuda.current_stream())
    guards = []
    def guarded(shape):
        size = 1
        for dim in shape:
            size *= dim
        storage = torch.full((size + 256,), 19, device='cuda', dtype=torch.uint8)
        guards.append(storage)
        return storage[128:-128].view(shape)
    expected = (guarded((m, 768)), guarded((padded, 24)))
    actual = (guarded((m, 768)), guarded((padded, 24)))
    def launch(fn, out):
        parents = (m + 255) // 256 * 6
        _run_compiled(fn, q.view(torch.int8).view(-1), w.w13.view(torch.int8).view(-1),
                      out[0].view(torch.int8).view(-1), out[1].view(-1), ss.view(-1),
                      w.s13.view(-1), m, 1536, 0, parents, torch.cuda.current_stream())
    launch(old, expected)
    launch(new, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(new, actual)
    for seed in [901, 902, 901]:
        q1, s1 = shared_quant(make_activation(m, seed=seed))
        q.copy_(q1)
        _run_compiled(sd._shuffle_scales(6144), s1.view(-1), ss.view(-1), m, torch.cuda.current_stream())
        launch(old, expected)
        for _ in range(2):
            actual[0].fill_(255)
            actual[1].fill_(255)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(row_scales(actual[1], m), row_scales(expected[1], m))
            assert all((t[:128] == 19).all() and (t[-128:] == 19).all() for t in guards)


def test_full_tile_epilogue_variants():
    assert '_early_' in sd._gemm1((4, 4), schedule='early').kernel_name
    assert '_early_' not in sd._gemm1((4, 4)).kernel_name
    assert 'single_tile' in sd._gemm1((4, 4), schedule='single').kernel_name
    assert '_early_' not in sd._gemm1((2, 2), schedule='early').kernel_name
    assert '_early_' not in sd._gemm1((2, 4), schedule='early').kernel_name


@pytest.mark.parametrize('m', [8193, 32769])
def test_early_epilogue_next_to_mfma(weights, m, monkeypatch):
    from tests.mfma_spinner import mfma_spinner
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert import shared_expert
    _, w = weights
    x = make_activation(m, seed=m)
    original = sd._gemm1
    with monkeypatch.context() as mp:
        mp.setattr(sd, '_gemm1', lambda tiles=(4, 4), **kw: original(tiles))
        expected = shared_expert(x, w)
    shared_expert(x, w)
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    stream = torch.cuda.Stream(priority=-1)
    stream.wait_stream(torch.cuda.current_stream())
    spinner(5_000_000, stream)
    outputs = [shared_expert(x, w) for _ in range(3)]
    torch.cuda.synchronize()
    assert all(torch.equal(y, expected) for y in outputs)


@pytest.mark.parametrize('m,schedule', [(1024, 'deferred'), (1025, 'single'),
    (8192, 'single'), (10752, 'single'), (10753, 'early'), (32768, 'early')])
def test_shared_gemm1_schedule(m, schedule):
    assert sd._gemm1_schedule_for(m) == schedule
    name = sd._gemm1((4, 4), schedule=schedule).kernel_name
    assert ('single_tile' in name) == (schedule == 'single')
    assert ('_early_' in name) == (schedule == 'early')
