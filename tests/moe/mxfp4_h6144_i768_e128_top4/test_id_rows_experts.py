# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""GEMM1 builds its 16-row blocks from topk_ids (gemm1_a4w4_id_rows): slot bytes equal the sorted path, the published
block map is canonical, the rows run in pair order; under skew, changed routing in replay, poison and MFMA load."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_small as ex
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from .workload import make_experts, make_activation, make_routing, require_gfx950
from .test_shared_expert import weights as shared_weights


@pytest.fixture(scope='module')
def experts():
    require_gfx950()
    return make_experts(seed=9411)[1]


def canonical(ids):
    """Block -> expert list and the rows (token | slot << 24, padding token M) in pair order."""
    m = ids.shape[0]
    values = ids.cpu().view(-1).tolist()
    blocks, rows = [], []
    for e in range(128):
        pairs = [(q // 4) | ((q % 4) << 24) for q, v in enumerate(values) if v == e]
        for b in range((len(pairs) + 15) // 16):
            blocks.append(e)
            chunk = pairs[b * 16:(b + 1) * 16]
            rows.extend(chunk + [m] * (16 - len(chunk)))
    return blocks, rows


def same(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))


def spy_tables(monkeypatch):
    """Capture the map and row table the id-rows host allocates (the 2nd/3rd int32 empties)."""
    seen = []
    real = torch.empty

    def empty(*shape, **kw):
        t = real(*shape, **kw)
        if kw.get('dtype') == torch.int32 and len(shape) == 1:
            seen.append(t)
        return t
    monkeypatch.setattr(ex.torch, 'empty', empty)
    return seen


@pytest.mark.parametrize('m', [16, 17, 24, 32, 33, 48, 57, 58, 64, 65, 66, 72, 80, 81, 88, 95, 96])
@pytest.mark.parametrize('kind', ['uniform', 'hot', 'four', 'one'])
def test_skew_changed_graph(experts, shared_weights, m, kind, monkeypatch):
    assert ex.id_rows_for(m)
    w13, w2, scales = experts
    x = make_activation(m, seed=9412 + m); xq = mxfp4_quant(x)
    wt, ids = make_routing(m, seed=9413 + m, skewed=kind == 'hot')
    if kind == 'four': ids.copy_(torch.arange(4, device='cuda', dtype=torch.int32).expand(m, 4))
    elif kind == 'one': ids.fill_(127)
    shared = (*shared_quant(x), shared_weights[1])
    args = (xq, w13, w2, scales, wt, ids)
    expected = ex.a4w4_experts(*args, inline=False, shared=shared)

    def call():
        with monkeypatch.context() as p:
            def fail(*a, **kw): raise AssertionError('id-rows consumers called the standalone sort')
            p.setattr(ex, 'moe_sort_a16w4', fail)
            seen = spy_tables(p)
            out = ex.a4w4_experts(*args, shared=shared, id_rows=True)
            return out, seen[0], seen[1]
    out0, tile, rows = call()
    torch.cuda.synchronize()
    assert same(out0, expected)
    blocks, rows_ref = canonical(ids)
    assert int(tile[-1].item()) == len(blocks)
    assert tile[:len(blocks)].cpu().tolist() == blocks
    assert rows[:len(rows_ref)].cpu().tolist() == rows_ref
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out, tile, rows = call()
    for change in (False, True):
        if change:
            ids.copy_(torch.randint(0, 128, (m, 4), device='cuda', dtype=torch.int32))
            ids[:, 1] = ids[:, 0]  # duplicate ids within a token
            xnew = mxfp4_quant(make_activation(m, seed=9512 + m))
            for dst, src in zip(xq, xnew): dst.copy_(src)
            expected = ex.a4w4_experts(*args, inline=False, shared=shared)
            blocks, rows_ref = canonical(ids)
        for _ in range(3):
            tile.fill_(-1); rows.fill_(-7)
            for t in (out if isinstance(out, tuple) else (out,)): t.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert same(out, expected)
            assert int(tile[-1].item()) == len(blocks)
            assert tile[:len(blocks)].cpu().tolist() == blocks
            assert rows[:len(rows_ref)].cpu().tolist() == rows_ref


@pytest.mark.parametrize('m', [16, 33, 64, 65, 80, 96])
def test_next_to_mfma(experts, shared_weights, m):
    from tests.mfma_spinner import mfma_spinner
    w13, w2, scales = experts
    x = make_activation(m, seed=9612 + m); xq = mxfp4_quant(x)
    wt, ids = make_routing(m, seed=9613 + m, skewed=True)
    shared = (*shared_quant(x), shared_weights[1])

    def call():
        return ex.a4w4_experts(xq, w13, w2, scales, wt, ids, shared=shared, id_rows=True)
    expected = call(); torch.cuda.synchronize()
    spinner = mfma_spinner(256); stream = torch.cuda.Stream(priority=-1)
    spinner(5_000_000, stream)
    actual = call(); torch.cuda.synchronize()
    assert same(actual, expected)


@pytest.mark.parametrize('m', [65, 80, 96])
@pytest.mark.parametrize('kind', ['uniform', 'one'])
def test_guarded_row_table_allocations(experts, shared_weights, m, kind, monkeypatch):
    """Guard every scratch / output allocation of the 512-slot path, duplicate ids included."""
    import math
    w13, w2, scales = experts
    x = make_activation(m, seed=9712 + m)
    xq = mxfp4_quant(x)
    wt, ids = make_routing(m, seed=9713 + m)
    if kind == 'one':
        ids.fill_(127)
    shared = (*shared_quant(x), shared_weights[1])
    args = (xq, w13, w2, scales, wt, ids)
    expected = ex.a4w4_experts(*args, inline=False, shared=shared)
    real_empty, guards = torch.empty, []

    def guarded(*shape, **kw):
        dims = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else shape
        dtype = kw.get('dtype', torch.float32)
        g = 128 // real_empty(0, dtype=dtype).element_size()
        base = real_empty(math.prod(dims) + 2 * g, **kw).fill_(93)
        guards.append((base, g))
        return base[g:-g].view(dims)

    with monkeypatch.context() as p:
        p.setattr(ex.torch, 'empty', guarded)
        actual = ex.a4w4_experts(*args, shared=shared, id_rows=True)
    torch.cuda.synchronize()
    assert same(actual, expected)
    assert guards
    for base, g in guards:
        assert bool((base[:g] == 93).all())
        assert bool((base[-g:] == 93).all())
