# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``top_p_mask`` above 64 rows (vLLM's one-program-per-row kernel) against ``reference.py``, bit for bit: LM-like
rows at temperatures 0.6-1.5, a heavy tail (Gumbel body), integer ties (boundary duplicates), -inf columns, flat
rows (the full-buffer search), p 1 rows, a constant row, an all -inf row; the three outlier paths (threshold on e,
Triton's division per element, the exact outlier mass); more rows than CUs; a caller's buffer; row stride; graph
replay with new inputs; the warm-up compiles the kernel and nothing compiles later. Bit-equality with vLLM's Triton
kernel needs vLLM, so it is checked outside these tests."""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.sampling import top_p_mask_f32_v200064 as tp
from amd_kernels.sampling.top_p_mask_f32_v200064 import V, top_p_mask, top_p_mask_reference
from amd_kernels.sampling.top_p_mask_f32_v200064 import host as tp_host
from amd_kernels.sampling.top_p_mask_f32_v200064 import kernel_rows


def require_gfx950():
    if not tp.available():
        pytest.skip("requires gfx950 with FlyDSL")


def lm_rows(n, gen, temp=1.0):
    x = torch.randn(n, V, generator=gen) * 2.0
    x.scatter_(1, torch.randint(0, V, (n, 64), generator=gen), torch.rand(n, 64, generator=gen) * 12 + 6)
    return (x.to(torch.bfloat16).float() / temp).cuda()


def case(name, n, gen):
    p = torch.full((n,), 0.95, device="cuda")
    if name == "lm":
        return lm_rows(n, gen), p
    if name == "cold":
        return lm_rows(n, gen, 0.6), p
    if name == "hot":
        return lm_rows(n, gen, 1.5), torch.rand(n, generator=gen).cuda() * 0.9 + 0.05
    if name == "tail":
        x = -torch.log(-torch.log(torch.rand(n, V, generator=gen).clamp(1e-12, 1 - 1e-7))) * 1.3
        x.scatter_(1, torch.randint(0, V, (n, 8), generator=gen), torch.rand(n, 8, generator=gen) * 10 + 8)
        return x.to(torch.bfloat16).float().cuda(), p
    if name == "ties":
        return torch.randint(-4, 5, (n, V), generator=gen).float().cuda(), torch.full((n,), 0.9, device="cuda")
    if name == "flat":
        return (torch.randn(n, V, generator=gen) * 0.5).cuda(), p
    if name == "masked":
        x = lm_rows(n, gen)
        x[:, ::3] = float("-inf")
        x[: n // 4, 1::3] = float("-inf")
        return x, p
    if name == "mixed":  # p 1 rows, a one-finite row, a constant row, an all -inf row, 100 equal maxima
        x = lm_rows(n, gen)
        p[::4] = 1.0
        x[1] = float("-inf")
        x[1, 1234] = 3.0
        x[2] = 0.0
        x[3] = float("-inf")
        x[5, :100] = 50.0
        return x, p
    raise ValueError(name)


def same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def bad_rows(a, b):
    return (a.view(torch.int32) != b.view(torch.int32)).any(1).nonzero().flatten().tolist()


@pytest.mark.parametrize("n", [65, 97, 300])
@pytest.mark.parametrize("name", ["lm", "cold", "hot", "tail", "ties", "flat", "masked", "mixed"])
def test_matches_reference(n, name):
    require_gfx950()
    if n == 300 and name not in ("lm", "mixed"):
        pytest.skip("more rows than CUs: two cases")
    gen = torch.Generator().manual_seed(n * 7 + len(name))
    x, p = case(name, n, gen)
    y = top_p_mask(x.clone(), p)
    r = top_p_mask_reference(x, p)
    torch.cuda.synchronize()
    assert same(y, r), (name, n, bad_rows(y, r))
    keep = p >= 1
    assert same(y[keep], x[keep])


@pytest.mark.parametrize("path", ["exact", "mass"])
def test_outlier_paths(path):
    """The per-element division and the exact outlier mass (the fallbacks of the threshold path) on every row."""
    require_gfx950()
    gen = torch.Generator().manual_seed(21)
    x = torch.cat([case("lm", 40, gen)[0], case("hot", 20, gen)[0], case("ties", 10, gen)[0],
                   case("mixed", 10, gen)[0]])
    p = torch.rand(80, generator=gen).cuda() * 0.9 + 0.05
    p[::7] = 1.0
    exe = kernel_rows.compile_top_p_mask_rows(path=path)
    buf = torch.empty(80, V, device="cuda")
    y = x.clone()
    flyc.compile(exe, y.data_ptr(), y.stride(0), p.data_ptr(), tp_host._sigma_table(y.device).data_ptr(),
                 buf.data_ptr(), 80, 80, torch.cuda.current_stream())
    r = top_p_mask_reference(x, p)
    torch.cuda.synchronize()
    assert same(y, r), bad_rows(y, r)


def test_buffer_and_row_stride():
    require_gfx950()
    gen = torch.Generator().manual_seed(22)
    x, p = case("lm", 70, gen)
    big = torch.full((70, V + 64), 7.0, device="cuda")
    big[:, :V] = x
    buf = torch.empty(tp.rows_grid(70, 0), V, device="cuda")
    top_p_mask(big[:, :V], p, buffer=buf)
    assert same(big[:, :V], top_p_mask(x.clone(), p)) and bool((big[:, V:] == 7.0).all())


def test_graph_replay_new_inputs():
    require_gfx950()
    gen = torch.Generator().manual_seed(23)
    n = 90
    x, p = case("lm", n, gen)
    buf = x.clone()
    work = torch.empty(tp.rows_grid(n, 0), V, device="cuda")
    tp.warmup_top_p_mask(torch.device("cuda"))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        top_p_mask(buf, p, buffer=work)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        top_p_mask(buf, p, buffer=work)
    for name in ("hot", "ties", "mixed", "flat"):
        x2, p2 = case(name, n, gen)
        buf.copy_(x2)
        p.copy_(p2)
        g.replay()
        torch.cuda.synchronize()
        assert same(buf, top_p_mask_reference(x2, p2)), name


def test_warm_up_compiles_rows_kernel():
    """``warmup_top_p_mask`` compiles the > 64-row kernel; calls at any row count compile nothing after it."""
    require_gfx950()
    from flydsl.compiler.jit_function import MlirCompiler

    tp.warmup_top_p_mask(torch.device("cuda"))
    assert torch.device("cuda", 0) not in tp_host._ROWS_WORKSPACE
    original = MlirCompiler.compile

    def unexpected(*a, **k):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        gen = torch.Generator().manual_seed(24)
        for n in (65, 128, 257):
            x, p = case("lm", n, gen)
            top_p_mask(x, p)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = original
