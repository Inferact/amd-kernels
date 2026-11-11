# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The one-GPU kernels (``add_rmsnorm``) against ``reference.py``: bf16 out, MXFP8 out with tiled codes; graph replay on
new inputs into NaN-filled outputs; three launches bitwise equal."""

import pytest
import torch
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    CODE_MISMATCH,
    MXFP8_COS,
    OUT_COS,
    H,
    OUT_MISMATCH,
    SCALE_MISMATCH,
    add_rmsnorm_reference,
    compare_mxfp8,
    compare_out,
    scale_offsets,
    unshuffle_scales,
    untile_codes,
)

EPS = 1e-6
# per-32-column magnitudes, 1e+-20 row scales (rows above ~2e17 overflow the fp32 sum of squares), bf16 subnormals,
# zero rows (graph padding), exact ties
KINDS = ["randn", "mixed", "rowscale", "tiny", "zero", "ties"]
CODE_FILL, SCALE_FILL = 0x5A, 0xA5


def require_gfx950():
    if not an.available():
        pytest.skip("requires gfx950 with FlyDSL")


def make_rows(m, gen, kind):
    """bf16 [m, H] on the GPU in one of the input kinds."""
    if kind == "mixed":  # per-32-column magnitudes 1e-3 .. 3e3, 2% zero groups
        g = torch.randn((m, H // 32, 32), generator=gen)
        mag = 10 ** (torch.rand((m, H // 32, 1), generator=gen) * 6.5 - 3)
        mag[torch.rand((m, H // 32, 1), generator=gen) < 0.02] = 0
        x = (g * mag).reshape(m, H)
    elif kind == "rowscale":
        x = torch.randn((m, H), generator=gen) * (
            10 ** (torch.rand((m, 1), generator=gen) * 40 - 20)
        )
    elif kind == "tiny":  # bf16 subnormals
        x = torch.randn((m, H), generator=gen) * 2.0**-128
    elif kind == "zero":
        x = torch.zeros((m, H))
    elif kind == "ties":
        x = torch.randint(-3, 4, (m, H), generator=gen).float() * 0.5
    else:
        x = torch.randn((m, H), generator=gen)
    return x.bfloat16().cuda()


def make_weight(gen):
    return make_rows(1, gen, "mixed")[0] * 0.1


def residual_for(m, gen, kind):
    return make_rows(m, gen, "zero" if kind == "zero" else "randn")


def check_out(name, out, y64):
    st = compare_out(out, y64)
    print(f"{name}: {st}")
    assert st["nonfinite_ok"], (name, st)
    assert st["bad_ulp"] == 0 and st["mismatch"] <= OUT_MISMATCH, (name, st)
    assert st["cos"] >= OUT_COS, (name, st)


def check_mxfp8(name, codes, scales, y64, bf16_input=False):
    st = compare_mxfp8(codes, scales, y64, bf16_input)
    print(f"{name}: {st}")
    assert st["scale_max_diff"] <= 1 and st["scale_mismatch"] <= SCALE_MISMATCH, (
        name,
        st,
    )
    assert st["code_mismatch"] <= CODE_MISMATCH and st["code_far"] == 0, (name, st)
    assert st["cos"] >= MXFP8_COS, (name, st)


def code_buffers(m, tile):
    """Sentinel-filled code / scale buffers for ``m`` rows (``tile`` 0: row-major codes)."""
    rows = -(-m // tile) * tile if tile else m
    q = torch.full((rows, H), CODE_FILL, dtype=torch.uint8, device="cuda")
    s = torch.full(((rows + 31) // 32 * 32, H // 32), SCALE_FILL, dtype=torch.uint8)
    return q, s.cuda()


def check_padding(name, q, s, m, tile):
    """The kernels write only the codes and scale bytes of the m real rows."""
    qm = torch.ones(q.numel(), dtype=torch.bool, device=q.device)
    if tile:
        r = torch.arange(m, device=q.device)[:, None]
        c = torch.arange(H, device=q.device)[None, :]
        off = r // 16 * (16 * H) + c // 64 * 1024 + c % 64 // 16 * 256 + r % 16 * 16 + c % 16
        qm[off.flatten()] = False
    else:
        qm[: m * H] = False
    sm = torch.ones(s.numel(), dtype=torch.bool, device=s.device)
    sm[scale_offsets(m, s.device).flatten()] = False
    assert torch.all(q.flatten()[qm] == CODE_FILL), f"{name}: codes padding written"
    assert torch.all(s.flatten()[sm] == SCALE_FILL), f"{name}: scale padding written"


def reference(x, r, w):
    return add_rmsnorm_reference(x.cpu(), r.cpu(), w.cpu(), EPS)


@pytest.mark.parametrize("m", [1, 7, 24, 64, 172, 240, 1024, 4099])
@pytest.mark.parametrize("kind", KINDS)
def test_add_rmsnorm_bf16(m, kind):
    require_gfx950()
    gen = torch.Generator().manual_seed(m)
    xb = torch.zeros((m, H + 256), dtype=torch.bfloat16, device="cuda")
    xb[:, 128 : 128 + H] = make_rows(m, gen, kind)
    x = xb[:, 128 : 128 + H]  # a strided view, as qkv / all-reduce outputs can be
    r = residual_for(m, gen, kind)
    w = make_weight(gen)
    out, res = an.gemma_add_rmsnorm(x, r, w, EPS)
    y, res_ref = reference(x, r, w)
    assert torch.equal(res.cpu().view(torch.int16), res_ref.view(torch.int16))
    check_out(f"bf16 m={m} {kind}", out, y)


# (m, code tile): the decode GEMM's tiles 16 / 64 / 96, partial last tiles
MXFP8_TILED = [(1, 16), (17, 16), (17, 96), (64, 64), (172, 64), (256, 16), (320, 96), (4099, 16)]


@pytest.mark.parametrize("m,tile", MXFP8_TILED)
@pytest.mark.parametrize("kind", KINDS)
def test_add_rmsnorm_mxfp8_tiled(m, tile, kind):
    require_gfx950()
    gen = torch.Generator().manual_seed(1000 + m)
    x, r = make_rows(m, gen, kind), residual_for(m, gen, kind)
    w = make_weight(gen)
    q, s = code_buffers(m, tile)
    q2, s2, res = an.gemma_add_rmsnorm_mxfp8(
        x, r, w, EPS, tile, q_out=q.view(torch.float8_e4m3fn), scale_out=s
    )
    assert q2.data_ptr() == q.data_ptr() and s2.data_ptr() == s.data_ptr()
    y, res_ref = reference(x, r, w)
    assert torch.equal(res.cpu().view(torch.int16), res_ref.view(torch.int16))
    name = f"mxfp8 tiled m={m} tile={tile} {kind}"
    check_mxfp8(name, untile_codes(q, m), unshuffle_scales(s, m), y)
    check_padding(name, q, s, m, tile)


def _all_paths(x, r, w, bufs):
    """Every kernel path once; returns the outputs (views of the buffers / fresh tensors from the wrappers)."""
    q1, s1 = bufs
    out, res = an.gemma_add_rmsnorm(x, r, w, EPS)
    _, _, res_q = an.gemma_add_rmsnorm_mxfp8(
        x, r, w, EPS, 64, q_out=q1.view(torch.float8_e4m3fn), scale_out=s1
    )
    return [out, res, q1, s1, res_q]


def _check_all(outs, x, r, w, m):
    out, res, q1, s1, res_q = outs
    y, res_ref = reference(x, r, w)
    bits = lambda t: t.cpu().view(torch.int16)  # noqa: E731
    for t in (res, res_q):
        assert torch.equal(bits(t), res_ref.view(torch.int16))
    check_out("replay bf16", out, y)
    check_mxfp8("replay mxfp8", untile_codes(q1, m), unshuffle_scales(s1, m), y)


@torch.inference_mode()
def test_graph_replay_new_inputs():
    """Captured once at a decode bucket; outputs filled with NaN / sentinels, new inputs copied in, replayed: every
    output matches the contract again."""
    require_gfx950()
    m = 64
    gen = torch.Generator().manual_seed(3)
    x, r, w = make_rows(m, gen, "mixed"), make_rows(m, gen, "randn"), make_weight(gen)
    bufs = code_buffers(m, 64)
    _all_paths(x, r, w, bufs)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = _all_paths(x, r, w, bufs)
    for it, kinds in enumerate([("mixed", "rowscale"), ("randn", "ties")]):
        x.copy_(make_rows(m, gen, kinds[0]))
        r.copy_(make_rows(m, gen, kinds[1]))
        for t in outs:
            t.view(torch.uint8).fill_(0xFF)  # bf16 NaN, e4m3 NaN, scale 255
        graph.replay()
        torch.cuda.synchronize()
        _check_all(outs, x, r, w, m)


@pytest.mark.parametrize("m", [172, 4099])
@torch.inference_mode()
def test_deterministic(m):
    """Three launches of every path on the same inputs are bitwise equal."""
    require_gfx950()
    gen = torch.Generator().manual_seed(9 + m)
    x, r, w = make_rows(m, gen, "mixed"), make_rows(m, gen, "randn"), make_weight(gen)
    runs = []
    for _ in range(3):
        outs = _all_paths(x, r, w, code_buffers(m, 64))
        runs.append([t.clone() for t in outs])
    torch.cuda.synchronize()
    for run in runs[1:]:
        for i, (a, b) in enumerate(zip(runs[0], run)):
            a8, b8 = a.view(torch.uint8), b.view(torch.uint8)
            assert torch.equal(a8, b8), (i, int((a8 != b8).sum()), (a8 != b8).nonzero()[:4].tolist())


def _stats_equal(a, b):
    """GPU vs CPU statistics: counts exact; the fp64 reductions (cos, rel_l2) to 1e-12 (summation order); ``mismatch``
    to 1e-4: torch's CPU fp64 -> bf16 goes through fp32 (two roundings, ~2^-16 of the values off by one bf16 ulp),
    the GPU rounds once (6 of 589824 values here, each the GPU's the exact RNE)."""
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and a[k] != a[k]:  # max_ulps / cos of an output with a NaN
            assert b[k] != b[k], (k, a[k], b[k])
        elif isinstance(a[k], float) and k in ("cos", "rel_l2"):
            assert abs(a[k] - b[k]) <= 1e-12, (k, a[k], b[k])
        elif k == "mismatch":
            assert abs(a[k] - b[k]) <= 1e-4, (k, a[k], b[k])
        else:
            assert a[k] == b[k], (k, a[k], b[k])


def test_compare_out_on_the_gpu_finds_what_the_cpu_finds():
    """compare_out on the GPU: the CPU's statistics, and it reports a value 2 ulp off and a NaN where the reference is
    finite."""
    require_gfx950()
    gen = torch.Generator().manual_seed(5)
    y = (torch.randn((96, H), generator=gen, dtype=torch.float64) * 3).cuda()
    y[7, :64] = 0.0
    y[9, 100] = float("inf")
    out = y.bfloat16()
    clean = compare_out(out, y)
    assert clean["bad_ulp"] == 0 and clean["mismatch"] == 0 and clean["nonfinite_ok"], clean
    bad = out.clone()
    bad[3, 5] = (bad[3, 5].double() * (1 + 2**-6)).bfloat16()  # two bf16 ulps up
    bad[11, 17] = float("nan")
    for o in (out, bad):
        _stats_equal(compare_out(o, y), compare_out(o.cpu(), y.cpu()))
    st = compare_out(bad, y)
    assert st["bad_ulp"] == 1 and not st["nonfinite_ok"] and st["mismatch"] > 0, st


def test_compare_mxfp8_on_the_gpu_finds_what_the_cpu_finds():
    """compare_mxfp8 on the GPU: the CPU's statistics, and it reports a code three magnitudes off and a scale off by
    two."""
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import mxfp8_codes, mxfp8_scale_reference

    require_gfx950()
    gen = torch.Generator().manual_seed(6)
    y = (torch.randn((64, H), generator=gen, dtype=torch.float64) * 2).cuda()
    sb = mxfp8_scale_reference(y)
    codes = mxfp8_codes(y, sb)
    clean = compare_mxfp8(codes, sb, y)
    assert clean["code_mismatch"] == 0 and clean["scale_max_diff"] == 0 and clean["code_far"] == 0, clean
    bad_codes, bad_sb = codes.clone(), sb.clone()
    c = int(bad_codes[2, 40])
    mag = c & 0x7F
    bad_codes[2, 40] = (c & 0x80) | (mag + 3 if mag + 3 <= 0x7E else mag - 3)  # three magnitudes off, not NaN
    bad_sb[5, 3] += 2
    for c, s in ((codes, sb), (bad_codes, bad_sb)):
        _stats_equal(compare_mxfp8(c, s, y), compare_mxfp8(c.cpu(), s.cpu(), y.cpu()))
    st = compare_mxfp8(bad_codes, bad_sb, y)
    assert st["code_far"] >= 1 and st["scale_max_diff"] == 2, st
