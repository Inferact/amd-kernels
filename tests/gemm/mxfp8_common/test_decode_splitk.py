# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The split-K decode kernel (``mxfp8_gemm_decode_splitk``) behind the ``decode_splitk`` rows: every row the
dispatcher serves fits one resident grid at every row count of its window, keeps its MFMA operands in VGPRs (final
ISA: no spill, no ``v_accvgpr_write``), meets the fp64 contract at its window edges through the bf16 and codes
entries (same bytes, bitwise repeats, graph replay) and compiles nothing after the warm-up; K split 3 gives the bytes
of the decode kernel's three K-waves on order-sensitive data (the QKV kw3 rows, so the QKV chain keeps its bytes);
the workgroups of one b % 8 run on one XCD (the partials stay in that XCD's L2), alone and next to a concurrent
kernel; calls on two streams at once and next to a concurrent kernel give the isolated bytes; the epoch counters stay
multiples of ``EPOCH``."""

import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common import mxfp8_gemm_decode_splitk as SK
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

from .test_gemm import require_gfx950

BF16 = torch.bfloat16
MAX_M = 512


def _shapes():
    hosts = set(G._SHAPE_HOSTS)
    return sorted(set(G.tuned_shapes(BF16)) | hosts)


def _rows():
    """{(n, k, config): [first M, last M]} of every split-K row the bf16 / codes entries select."""
    rows = {}
    for n, k in _shapes():
        for m in range(1, MAX_M + 1):
            for config in (G._select_config(m, n, k, BF16), G._select_prequant_config(m, n, k, BF16)):
                if config.kernel == G._DECODE_SPLITK_KERNEL:
                    lo, hi = rows.setdefault((n, k, config), [m, m])
                    rows[(n, k, config)] = [min(lo, m), max(hi, m)]
    return rows


ROWS = _rows()
SHAPES = sorted({(n, k) for n, k, _ in ROWS})


def test_rows_fit_one_resident_grid():
    assert {(4992, 6144), (6144, 4096), (4608, 6144), (12288, 6144), (6144, 6144), (3072, 6144)} <= set(SHAPES)
    for (n, k, config), (lo, hi) in ROWS.items():
        assert G._splitk_config_ok(n, k, config), config
        for m in range(lo, hi + 1):
            grid, n_mb, counters, _ = SK.geometry(n, m, config.tile_m // 16, config.tile_n // (16 * config.waves_m),
                                                  config.waves_m, config.split_k)
            assert grid <= SK.GRID_MAX and n_mb == -(-m // config.tile_m), (n, k, m, config)
            assert counters <= SK.GRID_MAX // 2 * 8 * SK.CNT_STRIDE
            assert G.prequant_code_tile_rows(m, n, k) == config.tile_m


def _launchers(shape):
    return [(cfg, G._splitk_launcher(n, k, cfg)) for (n, k, cfg) in sorted(ROWS, key=str) if (n, k) == shape]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"n{n}_k{k}" for n, k in SHAPES])
def test_rows_hold_operands_in_vgprs(shape, tmp_path):
    """Compiled in a subprocess (compile-only, cache off, ISA dumped)."""
    require_gfx950()
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, AKR_COMPILE_ONLY="1", FLYDSL_DEBUG_DUMP_ASM="1", FLYDSL_DUMP_DIR=str(tmp_path),
               FLYDSL_RUNTIME_ENABLE_CACHE="0")
    subprocess.run([sys.executable, "-m", "tests.gemm.mxfp8_common.test_decode_splitk", *map(str, shape)], cwd=root,
                   env=env, check=True)
    names = [exe.kernel_name for _, exe in _launchers(shape)]
    assert names
    for name in names:
        isa = glob.glob(f"{tmp_path}/{name}/*.s")
        assert isa, name
        text = Path(isa[0]).read_text()
        assert "v_accvgpr_write" not in text, name
        assert "scratch_" not in text and ".vgpr_spill_count: 0" in text, name


def _spread(shape, gen):
    """normal values with per-32 scales spread over +-5 binades"""
    rows, cols = shape
    e = torch.randint(-5, 6, (rows, cols // 32, 1), generator=gen, device="cuda")
    x = torch.randn(shape, generator=gen, device="cuda")
    return x * torch.exp2(e.expand(rows, cols // 32, 32).reshape(rows, cols).float())


_WEIGHTS = {}


def _spread_weight(n, k):
    if (n, k) not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        q, s = quantize_mxfp8((_spread((n, k), gen) * 0.02).bfloat16())
        _WEIGHTS[(n, k)] = (q, s, *G.preshuffle_mxfp8_weight(q, s))
    return _WEIGHTS[(n, k)]


EDGES = sorted({(n, k, m) for (n, k, _), (lo, hi) in ROWS.items() for m in (lo, hi)})


@pytest.mark.parametrize("n,k,m", EDGES, ids=[f"n{n}_k{k}_m{m}" for n, k, m in EDGES])
@torch.inference_mode()
def test_rows_contract(n, k, m):
    """Each row at its window edges: the fp64 contract per output row (a zero row), the bf16 and codes entries' same
    bytes, bitwise repeats, graph replay over a NaN-poisoned output and changed inputs."""
    require_gfx950()
    qw, sw, w, ws = _spread_weight(n, k)
    gen = torch.Generator(device="cuda").manual_seed(m + n)
    x = _spread((m, k), gen).bfloat16()
    x[m // 2].zero_()
    out = G.mxfp8_linear(x, w, ws, BF16)
    exact = mxfp8_gemm_reference(*quantize_mxfp8(x), qw, sw, accumulate=torch.float64).double()
    live = exact.norm(dim=1) > 0
    got = out.double()
    cos = torch.nn.functional.cosine_similarity(got[live], exact[live], dim=1)
    rel = (got[live] - exact[live]).norm(dim=1) / exact[live].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3 and not out[m // 2].any(), (float(cos.min()), float(rel.max()))
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, G.prequant_code_tile_rows(m, n, k))
    call = lambda: G.mxfp8_gemm_prequant(q, s, w, ws, BF16, m)  # noqa: E731
    for _ in range(3):
        assert torch.equal(call().view(torch.int16), out.view(torch.int16))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = call()
    captured.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), out.view(torch.int16))
    q.copy_(G.mxfp8_e4m3_quantize_shuffled(_spread((m, k), gen).bfloat16(), q.shape[0])[0])
    g.replay()
    eager = call()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("shape", SHAPES, ids=[f"n{n}_k{k}" for n, k in SHAPES])
@torch.inference_mode()
def test_no_compile_after_warmup(shape, monkeypatch):
    """After ``warmup_mxfp8_gemm_weight`` every row count of the split-K windows compiles nothing (bf16 and codes)."""
    from flydsl.compiler.jit_function import MlirCompiler

    require_gfx950()
    n, k = shape
    _, _, w, ws = _spread_weight(n, k)
    G.warmup_mxfp8_gemm_weight(w, ws, MAX_M, BF16)
    torch.cuda.synchronize()
    compiled = []
    real = MlirCompiler.compile

    def counting(self, *a, **kw):
        compiled.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(MlirCompiler, "compile", counting)
    x = torch.zeros((MAX_M, k), device="cuda", dtype=BF16)
    ms = sorted({m for (n_, k_, _), (lo, hi) in ROWS.items() if (n_, k_) == shape for m in range(lo, hi + 1)})
    for m in ms:
        G.mxfp8_linear(x[:m], w, ws, BF16)
        q, s = G.mxfp8_e4m3_quantize_shuffled(x[:m], G.prequant_code_tile_rows(m, n, k))
        G.mxfp8_gemm_prequant(q, s, w, ws, BF16, m)
    torch.cuda.synchronize()
    assert not compiled, len(compiled)


def _wide(shape, gen):
    """order-sensitive data: per-32 scales over +-12 binades (fp32 sums depend on the K order)"""
    rows, cols = shape
    e = torch.randint(-12, 13, (rows, cols // 32, 1), generator=gen, device="cuda")
    x = torch.randn(shape, generator=gen, device="cuda")
    return (x * torch.exp2(e.expand(rows, cols // 32, 32).reshape(rows, cols).float())).bfloat16()


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, s = quantize_mxfp8(_wide((n, k), gen) * 0.01)
    return q, s, *G.preshuffle_mxfp8_weight(q, s)


@pytest.mark.parametrize("m", [65, 96, 112, 113, 176, 256])
@torch.inference_mode()
def test_split3_is_the_kw3_decode_order(m):
    require_gfx950()
    n, k = 4992, 6144
    qw, sw, w, ws = _weight(n, k, 7)
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = _wide((m, k), gen)
    row = G._select_prequant_config(m, n, k, BF16)
    assert row.kernel == G._DECODE_SPLITK_KERNEL and row.split_k == 3
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, row.tile_m)
    got = G.mxfp8_gemm_shuffled(q, s, w, ws, BF16, m)
    kw3 = G._Config(48, 48, 128, 0, 0, 1, G._DECODE_KERNEL, 3, 0)
    q3, s3 = G.mxfp8_e4m3_quantize_shuffled(x, 48)
    ref = G.mxfp8_gemm_shuffled(q3, s3, w, ws, BF16, m, config=kw3)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    exact = mxfp8_gemm_reference(*quantize_mxfp8(x), qw, sw, accumulate=torch.float64).double()
    assert not torch.equal(exact.bfloat16().view(torch.int16), got.view(torch.int16))  # the data sees the K order


def _probe_launcher():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import T

    from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import _addr

    @fx.struct
    class Shared:
        pad: fx.Array[fx.Uint8, 150 * 1024, 16]

    @flyc.kernel(name="xcc_probe_512", known_block_size=[512, 1, 1])
    def kernel(arg_out: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        xcc = fx.Int32(llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s",
                                       has_side_effects=True))
        if tx == 0:
            smem.pad.ptr[0] = fx.Uint8(1)
            p = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
                            arg_out + fx.Int64(bx * 4))
            p[0] = xcc

    @flyc.jit
    def launch(OUT: fx.Tensor, grid: fx.Int32, stream: fx.Stream):
        kernel(_addr(OUT)).launch(grid=(fx.Int64(grid), 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def _background(stream, n=4):
    a = torch.randn(4096, 4096, device="cuda", dtype=BF16)
    small = torch.randn(333, 517, device="cuda", dtype=BF16)

    def run():
        with torch.cuda.stream(stream):
            for _ in range(n):
                a @ a
                for _ in range(8):
                    small @ small.T
    return run


@torch.inference_mode()
def test_one_xcd_per_residue():
    """XCC_ID of every workgroup of a 256 x 512-thread grid (one per CU, the kernel's shape): the workgroups of one
    b % 8 share an XCD in every launch, alone and next to a concurrent stream."""
    from amd_kernels.common.tensor_shim import _run_compiled

    require_gfx950()
    exe = _probe_launcher()
    side = torch.cuda.Stream()
    busy = _background(side)
    outs = []
    for i in range(400):
        if i % 20 == 0:
            busy()
        out = torch.full((256,), -1, dtype=torch.int32, device="cuda")
        _run_compiled(exe, out, 256, torch.cuda.current_stream())
        outs.append(out)
    torch.cuda.synchronize()
    b = torch.arange(256, device="cuda", dtype=torch.int32)
    for out in outs:
        assert (out >= 0).all()
        groups = out.view(32, 8)  # [b // 8, b % 8]
        assert (groups == groups[:1]).all(), out.tolist()
        assert len(set((out - b).remainder(8).tolist())) == 1


def _calls(n, k, m, seed):
    _, _, w, ws = _weight(n, k, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + m)
    x = _wide((m, k), gen)
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, G.prequant_code_tile_rows(m, n, k))
    return lambda: G.mxfp8_gemm_shuffled(q, s, w, ws, BF16, m)


CASES = [(4992, 6144, 96), (4992, 6144, 176), (6144, 4096, 96), (3072, 6144, 96), (12288, 6144, 160)]


@torch.inference_mode()
def test_two_streams_and_a_concurrent_kernel():
    require_gfx950()
    calls = [_calls(n, k, m, i) for i, (n, k, m) in enumerate(CASES)]
    ref = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    side = torch.cuda.Stream()
    busy = _background(side, 2)
    got = []
    for it in range(60):
        busy()
        for j, st in enumerate(streams):
            with torch.cuda.stream(st):
                for i, c in enumerate(calls):
                    if (i + j + it) % 2 == 0:
                        got.append((i, c().clone()))
    torch.cuda.synchronize()
    assert len(got) >= 300
    for i, out in got:
        assert torch.equal(out.view(torch.int16), ref[i].view(torch.int16)), CASES[i]
    for (_, stream), cnt in G._SPLITK_COUNTERS.items():
        assert (cnt % SK.EPOCH == 0).all(), stream


@torch.inference_mode()
def test_graph_replay_and_counters():
    require_gfx950()
    calls = [_calls(n, k, m, 10 + i) for i, (n, k, m) in enumerate(CASES)]
    ref = [c().clone() for c in calls]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [c() for c in calls for _ in range(3)][2::3]
    for o in outs:
        o.fill_(float("nan"))
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    for o, r in zip(outs, ref):
        assert torch.equal(o.view(torch.int16), r.view(torch.int16))
    for cnt in G._SPLITK_COUNTERS.values():
        assert (cnt % SK.EPOCH == 0).all() and cnt.any()


if __name__ == "__main__":
    from amd_kernels.common.tensor_shim import _run_compiled

    shape = (int(sys.argv[1]), int(sys.argv[2]))
    d = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    for config, exe in _launchers(shape):
        _run_compiled(exe, d, d, d, d, d, d, d, config.tile_m, 1024, 1, 8 * config.split_k,
                      torch.cuda.current_stream())
