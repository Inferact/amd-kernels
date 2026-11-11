# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``swa_decode_fp8kv`` against the contract in ``swa_decode_fp8kv_gqa16_d128/reference.py``.

Served shapes (README, "What it does"): TP4 (16 query heads, 1 KV head, 128-token
pages): spec3 = 4 rows per request, bf16 output; the DSpark draft's 7-row block, causal
and non-causal, with the MXFP8 epilogue for o_proj. Every case: each request and KV head
against the fp64 reference, three launches bit for bit equal, and a CUDA graph replayed
into NaN-filled outputs. The batches carry e4m3 NaN bytes in every slot past a sequence
end (the kernel must not read them) and, in the ragged cases, padding requests
(seq_len 0, no rows) as CUDA-graph padding does.
"""

import functools

import pytest
import torch
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import (
    MAX_QL,
    MAX_WINDOW,
    host,
    kernel,
    max_window,
    supports_swa_decode_fp8kv,
    swa_decode_fp8kv,
)
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128.reference import (
    D,
    HG,
    check_close,
    decode_reference,
    quant_buffers_reference,
)
from amd_kernels.gemm.mxfp8_common import gemm as G

W = 4096  # the M3.1 draft's sliding window
K_SCALE, V_SCALE = 0.75, 1.25
SM_SCALE = D**-0.5

# (id, requests, rows per request (int or a ragged cycle), KV heads, page, causal, padding requests)
CASES = [
    # TP4 spec3
    ("tp4_r4_p128_r1", 1, 4, 1, 128, True, 0),
    ("tp4_r4_p128_r41", 41, 4, 1, 128, True, 0),
    ("tp4_r4_p128_r64", 64, 4, 1, 128, True, 0),
    # TP4 DSpark 7 (served: 16 / 1 heads, 7 rows, 128-token pages, causal)
    ("tp4_r7_p128_r1", 1, 7, 1, 128, True, 0),
    ("tp4_r7_p128_r2", 2, 7, 1, 128, True, 0),
    ("tp4_r7_p128_r8", 8, 7, 1, 128, True, 0),
    ("tp4_r7_p128_r32", 32, 7, 1, 128, True, 0),
    ("tp4_r7_p128_r48", 48, 7, 1, 128, True, 0),
    ("tp4_r7_p128_r64", 64, 7, 1, 128, True, 0),
    ("tp4_r7_ragged_p128_r4", 4, (7, 1, 3, 2), 1, 128, True, 1),
    ("tp4_r7_ragged_p128_r7", 7, (7, 1, 3, 2), 1, 128, True, 3),
    ("tp4_r7_ragged_p128_r16", 16, (7, 1, 3, 2), 1, 128, True, 3),
    # three-row tiles: 1-row groups at every group index (rows 1, 4, 7), empty groups, padding
    ("tp4_r7_ragged_p128_r28", 28, (7, 1, 3, 2, 4, 5, 6), 1, 128, True, 3),
    ("tp4_r7_ragged_p128_r64", 64, (7, 1, 3, 2), 1, 128, True, 3),
    # TP4 DSpark 7, non-causal (upstream vLLM's DSpark draft block): the query tiles at every tier
    # (1-row 1-5, 2-row 6-10, 3-row 11-48 requests, request-wide from 49)
    ("tp4_r7_p128_noncausal_r1", 1, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r5", 5, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r6", 6, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r10", 10, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r11", 11, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r32", 32, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r48", 48, 7, 1, 128, False, 0),
    ("tp4_r7_p128_noncausal_r49", 49, 7, 1, 128, False, 0),
    ("tp4_r7_ragged_p128_noncausal_r4", 4, (7, 1, 3, 2), 1, 128, False, 1),
    ("tp4_r7_ragged_p128_noncausal_r7", 7, (7, 1, 3, 2), 1, 128, False, 3),
    ("tp4_r7_ragged_p128_noncausal_r28", 28, (7, 1, 3, 2, 4, 5, 6), 1, 128, False, 3),
    # ragged rows (a combining wave reused, a partial last row group) + padding requests
    ("ragged_781_p128_r9", 9, (7, 8, 1), 1, 128, False, 3),
]
OPROJ_N = 6144  # the TP4 o_proj (N 6144, K 2048)


def o_proj_tile(rows: int, k: int) -> int:
    """The code tile the o_proj's table row reads at ``rows`` rows (``gemm.prequant_code_tile_rows``)."""
    return G.prequant_code_tile_rows(rows, OPROJ_N, k, torch.bfloat16)


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")


def make_batch(reqs, ql, pad, gen, page, hkv, ctx_hi=9000):
    """Requests with distinct pages over their window; ``pad`` padding requests (no
    rows, seq 0) appended; e4m3 NaN bytes in every slot past a sequence end (the pool
    sized to the windows). Returns (cache,
    block_table, seq_lens, qsl, q, rows per request)."""
    qlens = torch.tensor(
        [ql] * reqs if isinstance(ql, int) else [ql[r % len(ql)] for r in range(reqs)]
    )
    ctx = torch.randint(0, ctx_hi, (reqs,), generator=gen)
    ctx[0] = ctx_hi + 3000  # one request with a full window
    seq = ctx + qlens
    n_pages = 16 + sum(
        int((s + page - 1) // page) - max(0, int(c) - W) // page for s, c in zip(seq, ctx)
    )
    nblk = int((seq.max() + page - 1) // page)
    bt = torch.zeros((reqs + pad, nblk), dtype=torch.int32)
    pool = (torch.randn((n_pages, hkv, page, 2 * D), generator=gen) * 1.2).to(
        torch.float8_e4m3fn
    )
    pool = pool.view(torch.uint8)
    perm = torch.randperm(n_pages, generator=gen)
    used = 0
    for r in range(reqs):
        nb = int((seq[r] + page - 1) // page)
        lo = max(0, int(ctx[r]) - W) // page
        bt[r, :nb] = torch.randint(0, n_pages, (nb,), generator=gen, dtype=torch.int32)
        bt[r, lo:nb] = perm[used : used + nb - lo].int()
        used += nb - lo
        last = int(bt[r, nb - 1])
        pool[last, :, int(seq[r]) - (nb - 1) * page :] = 0x7F  # e4m3 NaN
    assert used <= n_pages
    seq_lens = torch.cat([seq, torch.zeros(pad, dtype=seq.dtype)]).int()
    qsl = torch.cat([torch.zeros(1), qlens.cumsum(0), qlens.sum().repeat(pad)]).int()
    q = (torch.randn((int(qlens.sum()), HG * hkv, D), generator=gen) * 1.5).bfloat16()
    return [t.cuda() for t in (pool, bt, seq_lens, qsl, q)] + [qlens.tolist()]


@functools.cache
def scales():
    """Device scalars, made once (a CUDA graph cannot copy them in)."""
    return torch.tensor(K_SCALE, device="cuda"), torch.tensor(V_SCALE, device="cuda")


def launch(batch, max_ql, causal, out=None, quant_out=None, **kw):
    pool, bt, seq_lens, qsl, q, _ = batch
    ks, vs = scales()
    swa_decode_fp8kv(
        q, pool, bt, seq_lens, qsl, max_ql, SM_SCALE, W, causal, ks, vs, out,
        quant_out=quant_out, **kw,
    )  # fmt: skip


def spy_names(monkeypatch):
    """The kernel names ``host._get_kernel`` hands out."""
    names = []
    real = host._get_kernel

    def spy(*args):
        exe = real(*args)
        names.append(exe.kernel_name)
        return exe

    monkeypatch.setattr(host, "_get_kernel", spy)
    return names


def check_reference(batch, out, causal, name):
    pool, bt, seq_lens, qsl, q, qlens = batch
    hkv = pool.shape[1]
    ref = decode_reference(q, pool, bt, seq_lens, qsl, SM_SCALE, W, causal, K_SCALE, V_SCALE)
    starts = qsl.tolist()
    worst_cos, worst_rel = 1.0, 0.0
    for r in range(len(qlens)):
        lo, hi = starts[r], starts[r + 1]
        for h in range(hkv):
            sl = (slice(lo, hi), slice(HG * h, HG * (h + 1)))
            cos, rel = check_close(out[sl], ref[sl], f"{name} request {r} KV head {h}")
            worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    print(f"{name}: worst cosine {worst_cos:.7f}, worst rel L2 {worst_rel:.2e}")


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def graph_replay(fn, outs_fill):
    """Capture ``fn`` in a CUDA graph, refill the outputs, replay twice."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(2):
        outs_fill()
        graph.replay()
        torch.cuda.synchronize()
        yield


@pytest.mark.parametrize("name,reqs,ql,hkv,page,causal,pad", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_decode_contract(name, reqs, ql, hkv, page, causal, pad, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(reqs * 100 + hkv * 10 + page)
    batch = make_batch(reqs, ql, pad, gen, page, hkv)
    q = batch[4]
    max_ql = ql if isinstance(ql, int) else max(ql)
    names = spy_names(monkeypatch)
    outs = [torch.full_like(q, float("nan")) for _ in range(3)]
    for out in outs:
        launch(batch, max_ql, causal, out)
    torch.cuda.synchronize()
    # the dispatched kernel: rows, page, causal, the planned split count
    (kname,) = set(names)
    splits = host.launch_splits(q, batch[0], len(batch[2]))
    assert f"_r{max_ql}_" in kname and f"_s{splits}_w{W}_p{page}_" in kname, kname
    assert ("_c" in kname.split("_n", 1)[1]) == causal, kname
    assert same_bits(outs[0], outs[1]) and same_bits(outs[0], outs[2]), name
    check_reference(batch, outs[0], causal, name)

    out = outs[1]
    for _ in graph_replay(lambda: launch(batch, max_ql, causal, out), lambda: out.fill_(float("nan"))):
        assert same_bits(out, outs[0]), name


# the MXFP8 epilogue's served query block: 7 rows a request, 1 KV head, 128-token pages
QUANT_CASES = [c for c in CASES if c[2] == 7 and c[3] == 1]


@pytest.mark.parametrize("name,reqs,ql,hkv,page,causal,pad", QUANT_CASES, ids=[c[0] for c in QUANT_CASES])
@torch.inference_mode()
def test_decode_mxfp8_epilogue(name, reqs, ql, hkv, page, causal, pad):
    """The epilogue that writes o_proj's MXFP8 input (TP4 draft): its bytes are the reference
    quant of the bf16 output, in the code tile o_proj reads at this row count and
    row-major; rows past the requests' are not written; graph replay and repeats equal."""
    require_gfx950()
    gen = torch.Generator().manual_seed(reqs * 100 + hkv * 10 + page + 1)
    batch = make_batch(reqs, ql, pad, gen, page, hkv)
    q = batch[4]
    rows, k = q.shape[0], q.shape[1] * D
    out = torch.empty_like(q)
    launch(batch, ql, causal, out)
    for tile in sorted({o_proj_tile(rows, k), 0}):
        pad_rows = max(tile, 16)
        rows_p = (rows + pad_rows - 1) // pad_rows * pad_rows
        bufs = [
            (
                torch.full((rows_p, k), 0xA5, dtype=torch.uint8, device="cuda"),
                torch.full(((rows_p + 31) // 32 * 32, k // 32), 0x5A, dtype=torch.uint8, device="cuda"),
            )
            for _ in range(2)
        ]
        c_ref, s_ref = quant_buffers_reference(out.view(rows, k), *bufs[0], tile)
        for codes, scales_ in bufs:
            launch(batch, ql, causal, None, (codes, scales_, tile))
        torch.cuda.synchronize()
        for codes, scales_ in bufs:
            assert torch.equal(codes, c_ref), (name, tile, (codes != c_ref).sum().item())
            assert torch.equal(scales_, s_ref), (name, tile, (scales_ != s_ref).sum().item())
        codes, scales_ = bufs[1]

        def refill():
            codes.fill_(0xA5)
            scales_.fill_(0x5A)

        for _ in graph_replay(lambda: launch(batch, ql, causal, None, (codes, scales_, tile)), refill):
            assert torch.equal(codes, c_ref) and torch.equal(scales_, s_ref), (name, tile)


def test_supports_bounds():
    """The window bound per row count (64 page-lookup lanes per wave) and the shapes
    the kernel takes."""
    require_gfx950()
    q = torch.empty((8, 16, D), dtype=torch.bfloat16, device="cuda")
    cache = torch.empty((4, 1, 128, 2 * D), dtype=torch.uint8, device="cuda")
    assert MAX_WINDOW == 8156 and max_window(8) == 8154 and max_window(1) == 8156
    for ql in range(1, MAX_QL + 1):
        assert supports_swa_decode_fp8kv(q, cache, ql, W)
        assert supports_swa_decode_fp8kv(q, cache, ql, max_window(ql))
        assert not supports_swa_decode_fp8kv(q, cache, ql, max_window(ql) + 1)
    assert not supports_swa_decode_fp8kv(q, cache, MAX_QL + 1, W)
    assert not supports_swa_decode_fp8kv(q[:, :12], cache, 8, W)  # 12 query heads per KV head
    assert not supports_swa_decode_fp8kv(q, cache[:, :, :80], 8, W)  # page not a multiple of 32


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("reqs", [1, 4, 8, 16, 28, 32, 40, 64])
@torch.inference_mode()
def test_tp4_q7_batch_combine_next_to_mfma(reqs, causal):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    gen = torch.Generator().manual_seed(113 + reqs)
    batch = make_batch(reqs, 7, 0, gen, 128, 1)
    expected = torch.empty_like(batch[4])
    launch(batch, 7, causal, expected)
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)
        with torch.cuda.stream(main):
            out = torch.empty_like(expected)
            launch(batch, 7, causal, out)
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("reqs", [1, 8, 16, 32, 64])
@torch.inference_mode()
def test_tp4_q7_group_output_bytes(reqs, causal):
    require_gfx950()
    gen = torch.Generator().manual_seed(301 + reqs)
    batch = make_batch(reqs, 7, 0, gen, 128, 1)
    out = torch.empty_like(batch[4])
    launch(batch, 7, causal, out)
    rows, heads, dim = out.shape
    for tile in (0, 16):
        codes = torch.full(((rows + 31) // 32 * 32, heads * dim), 0xa5, dtype=torch.uint8, device="cuda")
        scales = torch.full(((rows + 31) // 32 * 32, heads * dim // 32), 0xa5, dtype=torch.uint8, device="cuda")
        launch(batch, 7, causal, None, (codes, scales, tile))
        wanted_c, wanted_s = quant_buffers_reference(out.view(rows, -1), codes, scales, tile)
        assert torch.equal(codes, wanted_c)
        assert torch.equal(scales, wanted_s)


@torch.inference_mode()
def test_tp4_q7_group_cross_batch_cosine():
    require_gfx950()
    gen = torch.Generator().manual_seed(471)
    batch = make_batch(1, 7, 0, gen, 128, 1)
    pool, bt, seq, qsl, q, qlens = batch
    expected = torch.empty_like(q)
    launch(batch, 7, True, expected)
    for reqs in (3, 6, 10, 16, 21, 32, 33, 41, 42, 64):
        other = (pool, bt.repeat(reqs, 1), seq.repeat(reqs),
                 torch.arange(reqs + 1, device=q.device, dtype=torch.int32) * 7,
                 q.repeat(reqs, 1, 1), [7] * reqs)
        out = torch.empty_like(other[4])
        launch(other, 7, True, out)
        a, b = out[:7].double(), expected.double()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
        rel = ((a-b).norm()/b.norm()).item()
        print(f"SWA q7 cross batch {reqs}: min head cos {cos:.9f}, rel L2 {rel:.4g}")
        assert cos >= 0.99999 and rel <= 2.5e-3


def test_layer_blocker():
    from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import layer_blocker

    for heads, kv_heads, window in ((16, 1, 4096), (16, 1, max_window(MAX_QL))):
        assert layer_blocker(128, heads, kv_heads, window) is None
    for args in ((64, 16, 1, 4096), (128, 8, 1, 4096), (128, 16, 2, 4096), (128, 16, 1, None), (128, 16, 1, 0),
                 (128, 16, 1, max_window(MAX_QL) + 1)):
        assert layer_blocker(*args) is not None, args
