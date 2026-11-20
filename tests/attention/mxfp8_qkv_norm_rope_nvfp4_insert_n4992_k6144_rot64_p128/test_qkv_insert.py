# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp8_qkv_norm_rope_nvfp4_insert`` at TP2 (N 4992, heads 32 / 2 / 2): bytes of the deferred chain at every tier,
the reference's tolerances, graph == eager, determinism, epoch counters back at a multiple of 16 after every call,
two streams at once, the routing of ``qkv.mxfp8_gemm_qkv_norm_rope_nvfp4_insert``.

Inputs: a synthetic MXFP8 weight (normal values, per-32 scales over +-5 binades), activations with per-32 scales over
+-5 binades and one zero row, serving slot runs (requests of up to 8 tokens from a random page offset, or one request
across a page boundary), the last token's slot -1, one index slot -1, pages full of stale random bytes, index-q a
strided view (row stride 320 bytes).
"""

import functools

import numpy as np
import pytest
import torch

from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128 import (
    host,
    mxfp8_qkv_norm_rope_nvfp4_insert,
    reference,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import qkv as chain_entry
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import reference as insert_ref
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import compare, quantize_mxfp8

N, K = 4992, 6144
KW = dict(num_heads=32, num_kv_heads=2, num_index_heads=2, eps=1e-6)
QW, IQW, IQ_ROW = 4096, 256, 320
# every one-launch row and the deferred chain's tier edges above it (split-K reduce-insert to 64, bf16 GEMM above)
EDGE = list(range(1, 17)) + [17, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257, 320, 321, 384, 385,
                             416, 448, 449, 512, 513, 576, 577, 624, 625, 672, 673, 720, 721, 768]


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def _spread(rng, shape):
    x = rng.standard_normal(shape, dtype=np.float32)
    e = rng.integers(-5, 6, size=(shape[0], shape[1] // 32)).astype(np.float32)
    return x * np.repeat(2.0 ** e, 32, axis=1)


@functools.cache
def weights():
    rng = np.random.default_rng(39)
    w = torch.from_numpy(_spread(rng, (N, K)) / 64).cuda().to(torch.bfloat16)
    wc, we = quantize_mxfp8(w)
    shuffled = g.preshuffle_mxfp8_weight(wc, we)
    norm = [torch.from_numpy(rng.standard_normal(128, dtype=np.float32) * 0.3).cuda().to(torch.bfloat16)
            for _ in range(4)]
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    f = torch.arange(131072, dtype=torch.float64)[:, None] * inv
    cs = torch.cat([f.cos(), f.sin()], 1).to(torch.bfloat16).cuda().contiguous()
    return (wc, we), shuffled, norm, cs


def make(m, seed):
    """One call's inputs: (shuffled codes, scales), (row-major codes, exponents), insert arguments."""
    (_, _), _, norm, cs = weights()
    rng = np.random.default_rng(seed)
    x = _spread(rng, (m, K))
    if m > 2:
        x[m // 2] = 0
    x = torch.from_numpy(x).cuda().to(torch.bfloat16)
    tile = g.prequant_code_tile_rows(m, N, K)
    codes, scales = g.mxfp8_e4m3_quantize_shuffled(x, tile)
    rc, re = quantize_mxfp8(x)
    blocks = max(m // 128 + 4, m // 8 + 1)
    sl = []
    if seed % 2:
        start = min(128 - (m % 128 or 128) // 2, 128 * 4 - m)
        sl = list(range(start, start + m))
    else:
        pages = rng.permutation(blocks)
        for i, length in enumerate([8] * (m // 8) + ([m % 8] if m % 8 else [])):
            start = int(rng.integers(0, 128 - length + 1))
            sl += [int(pages[i]) * 128 + start + j for j in range(length)]
    slots = torch.tensor(sl, device="cuda", dtype=torch.int64)
    islots = slots.clone()
    if m > 3:
        slots[m - 1] = -1  # at a run's end (inside a run is the insert's strict V-octet xfail)
    if m > 4:
        islots[3] = -1
    pos = torch.from_numpy(rng.integers(0, 131000, size=m)).cuda()
    kv = torch.from_numpy(rng.integers(0, 256, size=(blocks, 4, 128, 72), dtype=np.uint8)).cuda()
    idx = torch.from_numpy(rng.integers(0, 256, size=(blocks, 128, 72), dtype=np.uint8)).cuda()
    ks = torch.tensor(0.37, device="cuda")
    vs = torch.tensor(0.61, device="cuda")
    return (codes, scales), (rc, re), (*norm, cs, pos, slots, islots, kv, idx, ks, vs)


def outputs(m):
    q = torch.full((m, QW), 7, dtype=torch.bfloat16, device="cuda")
    iq_rows = torch.full((m, IQ_ROW), 5, dtype=torch.uint8, device="cuda")
    return q, iq_rows


def run(kind, m, shuffled_act, row_act, ins):
    """(q, index-q row bytes, kv, idx) of one call on fresh page copies."""
    _, (w, ws), _, _ = weights()
    q, iq_rows = outputs(m)
    iq = iq_rows.view(torch.float8_e4m3fn)[:, :IQW]
    args = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:])
    if kind == "entry":
        mxfp8_qkv_norm_rope_nvfp4_insert(*shuffled_act, w, ws, m, *args, q, iq, **KW)
    elif kind == "chain":  # the deferred chain
        qkv = g.mxfp8_gemm_shuffled(*shuffled_act, w, ws, torch.bfloat16, m, defer_reduce=True)
        if isinstance(qkv, g.SkPartials):
            splitk_reduce_norm_rope_nvfp4_insert(qkv.tmp, qkv.split_k, *args, q, iq, **KW)
        else:
            norm_rope_nvfp4_insert(qkv, *args, q, iq, **KW)
    else:
        (wc, we), _, _, _ = weights()
        reference.mxfp8_qkv_norm_rope_nvfp4_insert(*row_act, wc, we, *args, q, iq, **KW)
    torch.cuda.synchronize()
    return q, iq_rows, args[8], args[9]


def nbytes_differ(a, b):
    return [int((x.view(torch.uint8) != y.view(torch.uint8)).sum()) for x, y in zip(a, b)]


@pytest.mark.parametrize("m", EDGE)
@pytest.mark.parametrize("seed", [1, 2])
def test_bytes_equal_deferred_chain(m, seed):
    require_gfx950()
    sa, ra, ins = make(m, 1000 * m + seed)
    assert nbytes_differ(run("entry", m, sa, ra, ins), run("chain", m, sa, ra, ins)) == [0, 0, 0, 0]


@pytest.mark.parametrize("m", range(1, 17))
def test_v_octet_alignment_and_stale_bits(m):
    """Every run length and starting nibble, including full octets and page crossings; random stale page bytes expose
    masks that erase an unwritten nibble or scale byte. The final main slot stays -1; index slots skip on their own."""
    require_gfx950()
    sa, ra, ins = make(m, 6000 + m)
    for offset in range(8):
        slots = torch.arange(m, device="cuda", dtype=torch.int64) + 120 + offset
        islots = slots.clone()
        if m > 1:
            slots[-1] = -1
            islots[0] = -1
        args = (*ins[:6], slots, islots, *ins[8:])
        assert nbytes_differ(run("entry", m, sa, ra, args), run("chain", m, sa, ra, args)) == [0, 0, 0, 0]


@pytest.mark.parametrize("m", [1, 5, 8, 13, 16])
def test_reference_tolerance(m):
    """The one-launch rows against the contract (the deferred chain's rows hold their own packages' contracts)."""
    require_gfx950()
    sa, ra, ins = make(m, 77 * m + 1)
    got = run("entry", m, sa, ra, ins)
    ref = run("reference", m, sa, ra, ins)
    for name, a, b in (("q", got[0], ref[0]), ("index_q", got[1][:, :IQW].view(torch.float8_e4m3fn),
                                                  ref[1][:, :IQW].view(torch.float8_e4m3fn))):
        c = compare(a.float(), b.float())
        bound = reference.REL_L2 if name == "q" else reference.INDEX_Q_REL_L2
        assert c["cos"] >= reference.COS_MIN and c["rel_l2"] <= bound, (name, c)
    slots, islots = ins[6], ins[7]
    kv_mask, idx_mask = insert_ref.written_bits(ins[8], ins[9], slots, islots)
    for page, stale, mask in ((got[2], ins[8], kv_mask), (got[3], ins[9], idx_mask)):
        keep = mask.view(-1) == 0
        assert torch.equal(page.view(-1)[keep], stale.view(-1)[keep])
    gk, gks, gv, gvs = insert_ref.read_main_pages(got[2], slots)
    rk, rks, rv, rvs = insert_ref.read_main_pages(ref[2], slots)
    gi, gis = insert_ref.read_index_pages(got[3], islots)
    ri, ris = insert_ref.read_index_pages(ref[3], islots)
    for name, a, b in (("K", gk, rk), ("V", gv, rv), ("index_K", gi, ri)):
        assert (a != b).float().mean().item() <= reference.CODE_MISMATCH or a.numel() < 1 / reference.CODE_MISMATCH, name
    for a, b in ((gks, rks), (gvs, rvs), (gis, ris)):
        if a.numel():
            c = compare(a.view(torch.float8_e4m3fn).float(), b.view(torch.float8_e4m3fn).float())
            assert c["cos"] >= reference.COS_MIN and c["rel_l2"] <= reference.PAGE_REL_L2, c


def _all_counters_rearmed():
    return all(int((cnt & (host._kernel.EPOCH - 1)).count_nonzero()) == 0 for cnt in host._COUNTERS.values())


@pytest.mark.parametrize("m", [1, 4, 8, 9, 16, 17, 64, 256])
def test_graph_equals_eager_and_counters_rearm(m):
    """Outputs poisoned before every replay; after it the bytes equal the eager call's and every counter is a multiple
    of the epoch."""
    require_gfx950()
    _, (w, ws), _, _ = weights()
    calls = [make(m, 3000 * m + i) for i in range(4)]
    bufs = []
    for sa, ra, ins in calls:
        q, iq_rows = outputs(m)
        bufs.append((q, iq_rows, ins[8].clone(), ins[9].clone()))
    eager = [run("entry", m, sa, ra, ins) for sa, ra, ins in calls]
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for (sa, _, ins), (q, iq_rows, kv, idx) in zip(calls, bufs):
            mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *ins[:8], kv, idx, *ins[10:], q,
                                             iq_rows.view(torch.float8_e4m3fn)[:, :IQW], **KW)
    for _ in range(3):
        for (_, _, ins), (q, iq_rows, kv, idx) in zip(calls, bufs):
            q.fill_(1.5)
            iq_rows.fill_(0x33)
            kv.copy_(ins[8])
            idx.copy_(ins[9])
        gr.replay()
        torch.cuda.synchronize()
        for e, b in zip(eager, bufs):
            got = (b[0], torch.where(torch.arange(IQ_ROW, device="cuda") < IQW, b[1], e[1]), b[2], b[3])
            assert nbytes_differ(e, got) == [0, 0, 0, 0]
        assert _all_counters_rearmed()


@pytest.mark.parametrize("m", [1, 8, 13, 16])
def test_deterministic(m):
    require_gfx950()
    sa, ra, ins = make(m, 5 * m)
    first = run("entry", m, sa, ra, ins)
    for _ in range(5):
        assert nbytes_differ(first, run("entry", m, sa, ra, ins)) == [0, 0, 0, 0]


@pytest.mark.parametrize("ms", [(8, 5), (16, 1), (16, 64)])
def test_two_streams_at_once(ms):
    """Two streams (DBO) call the entry concurrently on their own inputs: each gets its own counters and bytes."""
    require_gfx950()
    _, (w, ws), _, _ = weights()
    jobs = [make(m, 900 + m) for m in ms]
    want = [run("entry", m, *job) for m, job in zip(ms, jobs)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    bufs = []
    for m, (sa, ra, ins) in zip(ms, jobs):
        q, iq_rows = outputs(m)
        bufs.append((q, iq_rows, ins[8].clone(), ins[9].clone()))
    torch.cuda.synchronize()
    for _ in range(20):
        for st, m, (sa, _, ins), (q, iq_rows, kv, idx) in zip(streams, ms, jobs, bufs):
            with torch.cuda.stream(st):
                kv.copy_(ins[8])
                idx.copy_(ins[9])
                mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *ins[:8], kv, idx, *ins[10:], q,
                                                 iq_rows.view(torch.float8_e4m3fn)[:, :IQW], **KW)
        torch.cuda.synchronize()
        for e, b in zip(want, bufs):
            assert nbytes_differ(e, b) == [0, 0, 0, 0]


def test_zero_rows():
    require_gfx950()
    sa, ra, ins = make(4, 11)
    _, (w, ws), _, _ = weights()
    q = torch.empty((0, QW), dtype=torch.bfloat16, device="cuda")
    iq = torch.empty((0, IQW), dtype=torch.float8_e4m3fn, device="cuda")
    kv = ins[8].clone()
    mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, 0, *ins[:5], ins[5][:0], ins[6][:0], ins[7][:0], kv, ins[9],
                                     *ins[10:], q, iq, **KW)
    torch.cuda.synchronize()
    assert torch.equal(kv, ins[8])


def test_rows_past_the_shuffled_tile_raise():
    sa, ra, ins = make(8, 12)
    _, (w, ws), _, _ = weights()
    q, iq_rows = outputs(769)
    with pytest.raises(ValueError):
        mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, 769, *ins, q, iq_rows.view(torch.float8_e4m3fn)[:, :IQW], **KW)


@pytest.mark.parametrize("m", [8, 16])
def test_next_to_mfma(m):
    """The hand-off and its packed arithmetic stay correct while another stream issues MFMAs."""
    require_gfx950()
    from tests.mfma_spinner import mfma_spinner

    sa, ra, ins = make(m, 7100 + m)
    want = run("chain", m, sa, ra, ins)
    spin = mfma_spinner(64)
    high = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    for _ in range(5):
        spin(200000, high)  # 2 ms, enough to overlap the entry on another stream
        assert nbytes_differ(want, run("entry", m, sa, ra, ins)) == [0, 0, 0, 0]
        assert _all_counters_rearmed()


def test_graphs_captured_on_one_stream_replay_concurrently():
    """A graph's capture stream is not necessarily its replay stream; capture IDs keep their counters separate."""
    require_gfx950()
    _, (w, ws), _, _ = weights()
    ms = (8, 16)
    jobs = [make(m, 8100 + m) for m in ms]
    want = [run("chain", m, *job) for m, job in zip(ms, jobs)]
    for m, job in zip(ms, jobs):
        run("entry", m, *job)  # pool initialization and compilation happen outside capture
    capture_stream = torch.cuda.Stream()
    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, bufs, pointers = [], [], []
    for m, (sa, _, ins) in zip(ms, jobs):
        q, iq_rows = outputs(m)
        kv, idx = ins[8].clone(), ins[9].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            pointers.append(host._counters(sa[0].device).data_ptr())
            mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *ins[:8], kv, idx, *ins[10:], q,
                                             iq_rows.view(torch.float8_e4m3fn)[:, :IQW], **KW)
        graphs.append(graph)
        bufs.append((q, iq_rows, kv, idx))
    assert pointers[0] != pointers[1]  # fail before launching unsafe shared counters
    for _ in range(20):
        for (_, _, ins), (q, iq_rows, kv, idx) in zip(jobs, bufs):
            q.fill_(7)
            iq_rows.fill_(5)
            iq_rows[:, :IQW].fill_(0x77)
            kv.copy_(ins[8])
            idx.copy_(ins[9])
        torch.cuda.synchronize()
        for _ in range(3):
            for stream, graph in zip(replay_streams, graphs):
                with torch.cuda.stream(stream):
                    graph.replay()
        torch.cuda.synchronize()
        assert all(nbytes_differ(a, b) == [0, 0, 0, 0] for a, b in zip(want, bufs))
        assert _all_counters_rearmed()


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("initial", [0x7FFFFFF0, -16])
def test_epoch_counter_wrap(m, initial):
    """Cross the signed and unsigned i32 boundaries without losing an arrival or changing output bytes; every head's
    counter (one per 128-byte line) takes exactly one epoch per call."""
    require_gfx950()
    sa, ra, ins = make(m, 9800 + m)
    want = run("chain", m, sa, ra, ins)
    run("entry", m, sa, ra, ins)
    cnt = host._counters(sa[0].device)
    cnt.fill_(initial)
    before = cnt.clone()
    epoch = host._kernel.EPOCH
    lines = torch.arange(cnt.numel(), device="cuda") % host._kernel.CNT_STRIDE == 0
    for iteration in range(2):
        got = run("entry", m, sa, ra, ins)
        assert nbytes_differ(want, got) == [0, 0, 0, 0]
        delta = (cnt.to(torch.int64) - before.to(torch.int64)) & 0xFFFFFFFF
        assert torch.equal(delta[lines], torch.full_like(delta[lines], epoch * (iteration + 1)))
        assert int(delta[~lines].count_nonzero()) == 0
        assert int((cnt & (epoch - 1)).count_nonzero()) == 0


@pytest.mark.parametrize("which", [10, 11])  # k_scale, v_scale
@pytest.mark.parametrize("m", [8, 16])
def test_graph_reads_changed_scale(m, which):
    """A captured call rebuilds its exact reciprocal tables from the current K / V scale."""
    require_gfx950()
    sa, ra, ins = make(m, 19500 + m)
    _, (w, ws), _, _ = weights()
    run("entry", m, sa, ra, ins)
    q, iq_rows = outputs(m)
    kv, idx = ins[8].clone(), ins[9].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mxfp8_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, *ins[:8], kv, idx, *ins[10:], q,
                                         iq_rows.view(torch.float8_e4m3fn)[:, :IQW], **KW)
    for scale in (0.61, 1.5, 2.0**-8, 2.0**8, 0.37):
        ins[which].fill_(scale)
        want = run("chain", m, sa, ra, ins)
        q.fill_(-7)
        iq_rows.fill_(5)
        iq_rows[:, :IQW].fill_(0x77)
        kv.copy_(ins[8])
        idx.copy_(ins[9])
        graph.replay()
        torch.cuda.synchronize()
        assert nbytes_differ((q, iq_rows, kv, idx), want) == [0, 0, 0, 0]


@pytest.mark.parametrize("m", [1, 8, 16, 17, 64, 65, 256, 416, 768])
@pytest.mark.parametrize("one_launch", [True, False])
def test_qkv_entry_routes_tp2_heads_here(m, one_launch):
    """``qkv.mxfp8_gemm_qkv_norm_rope_nvfp4_insert`` at N 4992: this entry where ``one_launch`` allows it, else the
    GEMM and the insert; the same bytes either way."""
    require_gfx950()
    sa, ra, ins = make(m, 4400 + m)
    _, (w, ws), _, _ = weights()
    assert chain_entry._one_launch_entry(N, K) is host
    want = run("chain", m, sa, ra, ins)
    kv, idx = ins[8].clone(), ins[9].clone()
    q, iq = chain_entry.mxfp8_gemm_qkv_norm_rope_nvfp4_insert(*sa, w, ws, m, torch.bfloat16, *ins[:8], kv, idx,
                                                             *ins[10:], **KW, one_launch=one_launch)
    torch.cuda.synchronize()
    assert nbytes_differ((q, iq, kv, idx), (want[0], want[1][:, :IQW], want[2], want[3])) == [0, 0, 0, 0]


@pytest.mark.parametrize("m", [17, 24, 32, 33, 40, 48])
def test_one_launch_past_its_window(m):
    """The one-launch kernel on the deferred split rows the host leaves to two launches (17-48): the same bytes."""
    require_gfx950()
    sa, ra, ins = make(m, 5100 + m)
    _, (w, ws), _, _ = weights()
    q, iq_rows = outputs(m)
    args = (*ins[:8], ins[8].clone(), ins[9].clone(), *ins[10:], q, iq_rows.view(torch.float8_e4m3fn)[:, :IQW])
    host._fused(*sa, w, ws, m, *args, KW["eps"])
    torch.cuda.synchronize()
    assert nbytes_differ((q, iq_rows, args[8], args[9]), run("chain", m, sa, ra, ins)) == [0, 0, 0, 0]
