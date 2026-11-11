# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The norm / RoPE / NVFP4 insert kernels against ``reference.py`` on the served heads, q16 kv1 iq1 (TP4): per path
(main kernel, prefill variant, split-K form) the contract, CUDA graph replay on new inputs, three bitwise-equal
repeats, and the variants' bytes equal to the main kernel's."""

import pytest
import torch
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    host,
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import reference as ref

D, BLK, WIDTH = ref.D, ref.BLK, ref.WIDTH
EPS = 1e-6
ROT = 64
HEADS = {"q16_kv1_iq1": (16, 1, 1)}
TP4 = HEADS["q16_kv1_iq1"]
DSPARK = ([3 + 9 * i for i in range(43)], [4] * 43)  # 172 decode rows (4 per request)
# (starts, lens, pad)
CASES = {
    "decode1": ([5], [1], 0),  # one decode token
    "octet_edge": ([0, 13], [7, 4], 1),  # a run of 7, runs across an octet edge
    "page_edges": ([125, 254, 4091], [6, 6, 6], 12),  # 6-token steps across pages
    "dspark172_pad20": (*DSPARK, 20),  # DSpark decode rows, graph padding
    "chunks300": ([0, 3, 250, 128], [131, 61, 9, 99], 0),  # unaligned prefill chunks
    "prefill8099": ([0, 100], [4096, 4000], 3),  # the 4-pairs tier
}


def prefill_case(n, n_req):
    """n tokens of n_req requests continuing ~40K-token contexts (starts off page / octet edges), the last 37
    tokens padding."""
    per = n // n_req
    starts = [40000 + 37 * r + 11 + (r * r) % 13 for r in range(n_req)]
    lens = [per] * (n_req - 1) + [n - 37 - per * (n_req - 1)]
    return starts, lens, 37


PREFILL_CASES = {f"n{n}_req{r}": prefill_case(n, r) for n, r in
                 [(4097, 3), (8191, 9), (16384, 9), (12345, 53)]}  # fmt: skip


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")


def request_slots(starts, lens, num_pages, gen):
    """The slot mapping vLLM builds: request r's tokens at positions starts[r] .. of
    its own pages (a random page order), requests back to back."""
    perm = torch.randperm(num_pages, generator=gen)
    out, p = [], 0
    for st, n in zip(starts, lens):
        npg = (st + n + BLK - 1) // BLK
        pos = torch.arange(st, st + n)
        out.append(perm[p : p + npg][pos // BLK] * BLK + pos % BLK)
        p += npg
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int64)


class Batch:
    """One step's inputs and stale pages. Values: magnitudes 1e-3 .. 3e3 per 16-value
    group and ~2% all-zero groups; positions are the
    tokens' own; index slots map the same tokens to other pages, ``index_skip`` of
    them unmapped; pages hold random bytes and have a block stride past one page."""

    def __init__(self, starts, lens, pad, heads, scales=(1.0, 1.0), seed=0, index_skip=0.0):
        g = torch.Generator().manual_seed(seed)
        self.nq, self.nkv, self.niq = heads
        self.n = n = sum(lens) + pad
        self.row = (self.nq + 2 * self.nkv + self.niq + 1) * D
        x = torch.randn(n, self.row // 16, 16, generator=g)
        mags = torch.tensor([1e-3, 0.05, 1.0, 1.0, 30.0, 3e3])
        x *= mags[torch.randint(0, len(mags), (n, self.row // 16), generator=g)][..., None]
        x[torch.rand(n, self.row // 16, generator=g) < 0.02] = 0.0
        self.qkv = x.view(n, self.row).to(torch.bfloat16).cuda()
        self.w = [(torch.randn(D, generator=g) * 0.5).to(torch.bfloat16).cuda() for _ in range(4)]
        pos = [torch.arange(s, s + m) for s, m in zip(starts, lens)] + [torch.zeros(pad, dtype=torch.int64)]
        pos = torch.cat(pos)
        rows = int(pos.max()) + 1 if n else 1
        self.cs = (torch.rand(rows + 64, ROT, generator=g) * 2 - 1).to(torch.bfloat16).cuda()
        self.pos = pos.cuda()
        pages = sum((s + m + BLK - 1) // BLK for s, m in zip(starts, lens)) + 2
        pad_slots = torch.full((pad,), -1, dtype=torch.int64)
        self.slots = torch.cat([request_slots(starts, lens, pages, g), pad_slots]).cuda()
        islots = torch.cat([request_slots(starts, lens, pages, g), pad_slots])
        islots[torch.rand(n, generator=g) < index_skip] = -1
        self.islots = islots.cuda()
        self.main = 2 * self.nkv * BLK * WIDTH
        self.kv_full = torch.randint(0, 256, (pages, self.main + 256), generator=g).to(torch.uint8).cuda()
        self.idx_full = torch.randint(0, 256, (pages, BLK * WIDTH + 64), generator=g).to(torch.uint8).cuda()
        self.ks = torch.tensor(scales[0], dtype=torch.float32, device="cuda")
        self.vs = torch.tensor(scales[1], dtype=torch.float32, device="cuda")

    def kv(self, full):
        return full[:, : self.main].view(-1, 2 * self.nkv, BLK, WIDTH)

    def idx(self, full):
        return full[:, : BLK * WIDTH].view(-1, BLK, WIDTH)

    def outputs(self):
        """Fresh outputs: q_out NaN, index_q_out 0x7F, the stale pages."""
        q = torch.full((self.n, self.nq * D), float("nan"), dtype=torch.bfloat16, device="cuda")
        iq = torch.full((self.n, self.niq * D), 0x7F, dtype=torch.uint8, device="cuda")
        return q, iq.view(torch.float8_e4m3fn), self.kv_full.clone(), self.idx_full.clone()

    def args(self, outs):
        q, iq, kv_full, idx_full = outs
        return (*self.w, self.cs, self.pos, self.slots, self.islots, self.kv(kv_full),
                self.idx(idx_full), self.ks, self.vs, q, iq)  # fmt: skip

    def kwargs(self):
        return dict(num_heads=self.nq, num_kv_heads=self.nkv, num_index_heads=self.niq, eps=EPS)

    def run(self, op=norm_rope_nvfp4_insert, prefill=False, outs=None):
        outs = self.outputs() if outs is None else outs
        qkv = self.qkv.clone()
        op(qkv, *self.args(outs), **self.kwargs(), prefill=prefill)
        assert torch.equal(qkv.view(torch.int16), self.qkv.view(torch.int16)), "qkv written"
        return outs

    def run_reduce(self, slab, op=splitk_reduce_norm_rope_nvfp4_insert, outs=None):
        outs = self.outputs() if outs is None else outs
        op(slab, slab.shape[0], *self.args(outs), **self.kwargs())
        return outs

    def reference(self, slab=None):
        outs = self.outputs()
        if slab is None:
            ref.norm_rope_nvfp4_insert(self.qkv, *self.args(outs), **self.kwargs())
        else:
            ref.splitk_reduce_norm_rope_nvfp4_insert(slab, slab.shape[0], *self.args(outs), **self.kwargs())
        return outs


def cos_rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    nb = b.norm().item()
    if a.numel() == 0 or nb == 0:
        return 1.0, float((a - b).norm())
    cos = (a @ b).item() / max(a.norm().item() * nb, 1e-300)
    return cos, (a - b).norm().item() / nb


def check_close(name, got, want, rel_max):
    cos, rel = cos_rel(got, want)
    print(f"{name}: cos 1-{1 - cos:.1e}, rel L2 {rel:.1e}")
    assert cos >= ref.COS_MIN and rel <= rel_max, (name, cos, rel)


def check(batch, got, want, tag=""):
    """``got`` against the reference outputs ``want`` (module docstring of reference.py)."""
    q, iq, kv, idx = got
    q_r, iq_r, kv_r, idx_r = want
    check_close(f"{tag} q_out", q.float(), q_r.float(), ref.REL_L2)
    check_close(f"{tag} index_q_out", iq.float(), iq_r.float(), ref.REL_L2)
    # every bit the reference does not write keeps the stale page
    kv_mask, idx_mask = ref.written_bits(batch.kv(kv), batch.idx(idx), batch.slots, batch.islots)
    for name, full, stale, mask, view in (
        ("kv", kv, batch.kv_full, kv_mask, batch.kv),
        ("index", idx, batch.idx_full, idx_mask, batch.idx),
    ):
        flat, flat_stale = ref.flat_pages(view(full)), ref.flat_pages(view(stale))
        changed = (flat ^ flat_stale) & ~mask
        assert not bool(changed.any()), f"{tag} {name}: {int((changed != 0).sum())} unwritten bytes changed"
        assert torch.equal(full[:, flat.shape[1]:], stale[:, flat.shape[1]:]), f"{tag} {name}: past the page"
    # written codes, dequantized
    if bool((batch.slots >= 0).any()):
        kc, ksf, vc, vsf = ref.read_main_pages(batch.kv(kv), batch.slots)
        kc_r, ksf_r, vc_r, vsf_r = ref.read_main_pages(batch.kv(kv_r), batch.slots)
        for side, c, s, c_r, s_r, gs in (("K", kc, ksf, kc_r, ksf_r, batch.ks), ("V", vc, vsf, vc_r, vsf_r, batch.vs)):
            check_close(f"{tag} {side} pages", ref.nvfp4_dequantize(c, s, gs), ref.nvfp4_dequantize(c_r, s_r, gs), ref.PAGE_REL_L2)
            miss = (c.long() != c_r.long()).float().mean().item()
            print(f"{tag} {side} codes: mismatch {miss:.1e}")
            assert miss <= ref.CODE_MISMATCH, (tag, side, miss)
    if bool((batch.islots >= 0).any()):
        ic, isf = ref.read_index_pages(batch.idx(idx), batch.islots)
        ic_r, isf_r = ref.read_index_pages(batch.idx(idx_r), batch.islots)
        one = torch.ones((), device="cuda")
        check_close(f"{tag} index-K pages", ref.nvfp4_dequantize(ic, isf, one), ref.nvfp4_dequantize(ic_r, isf_r, one), ref.PAGE_REL_L2)
        assert (ic.long() != ic_r.long()).float().mean().item() <= ref.CODE_MISMATCH, tag


def assert_bitwise(a_outs, b_outs, tag=""):
    for name, a, b in zip(("q_out", "index_q_out", "kv", "index"), a_outs, b_outs):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{tag} {name}"


def make_slab(batch, split_k, seed):
    torch.manual_seed(seed)
    return torch.randn(split_k, batch.n, batch.row, device="cuda")


# --- the contract --------------------------------------------------------------------


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.002, 3.0)], ids=["gs1", "gs_k0.002_v3"])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("heads", list(HEADS))
@torch.inference_mode()
def test_main_kernel_contract(heads, case, scales):
    """The main kernel (1 token pair per wave up to 512 tokens, 4 above)."""
    require_gfx950()
    batch = Batch(*CASES[case], HEADS[heads], scales)
    check(batch, batch.run(), batch.reference(), f"{heads} {case} pairs {host._token_pairs_per_wave(batch.n)}")


@pytest.mark.parametrize("m", [512, 513, 1024, 2048, 4095])
@torch.inference_mode()
def test_streaming_q_store_contract(m):
    """Four-pair q16 stores cover odd rows, a padding tail and stale cache pages."""
    require_gfx950()
    batch = Batch([125], [m - 1], 1, TP4, (0.5, 2.0), seed=m, index_skip=0.01)
    got = batch.run()
    check(batch, got, batch.reference(), f"streaming Q stores {m}")
    if m == 512:
        assert_bitwise(got, batch.run(prefill=True), "512-token prefill hint")


def test_insert_blocker():
    """The served layer fits; another head set, head dim, RoPE, index-q dtype or cos / sin cache gets a reason, the
    warm-up compiles nothing for it and the entries assert it (host only)."""
    assert host.insert_blocker(*TP4, ROT) is None
    cache = torch.zeros(8, ROT, dtype=torch.bfloat16)
    assert host.insert_blocker(*TP4, ROT, head_dim=D, index_head_dim=D, neox=True,
                               index_q_dtype=torch.float8_e4m3fn, cos_sin_cache=cache) is None
    for args, kw, word in [
        ((64, 4, 4, ROT), {}, "heads"),
        ((32, 2, 2, ROT), {}, "heads"),
        ((16, 1, 4, ROT), {}, "heads"),
        ((*TP4, 32), {}, "RoPE"),
        ((*TP4, ROT), dict(neox=False), "NeoX"),
        ((*TP4, ROT), dict(head_dim=64), "head dim"),
        ((*TP4, ROT), dict(index_q_dtype=torch.bfloat16), "index-q"),
        ((*TP4, ROT), dict(cos_sin_cache=cache.float()), "cos / sin"),
        ((*TP4, ROT), dict(cos_sin_cache=torch.zeros(8, 2 * ROT, dtype=torch.bfloat16)[:, :ROT]), "cos / sin"),
    ]:
        why = host.insert_blocker(*args, **kw)
        assert why is not None and word in why, (args, kw, why)
    why = host.insert_blocker(64, 4, 4, ROT)
    assert host.warmup_prefill_kernels(64, 4, 4, device=torch.device("cpu")) == why
    nq, nkv, niq = 64, 4, 4
    qkv = torch.zeros(1, (nq + 2 * nkv + niq + 1) * D, dtype=torch.bfloat16)
    w = torch.zeros(D, dtype=torch.bfloat16)
    one = torch.ones(())
    slots = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(AssertionError, match="heads per rank"):
        norm_rope_nvfp4_insert(
            qkv, w, w, w, w, torch.zeros(8, ROT, dtype=torch.bfloat16), slots, slots, slots,
            torch.zeros(1, 2 * nkv, BLK, WIDTH, dtype=torch.uint8), torch.zeros(1, BLK, WIDTH, dtype=torch.uint8),
            one, one, torch.zeros(1, nq * D, dtype=torch.bfloat16),
            torch.zeros(1, niq * D, dtype=torch.float8_e4m3fn),
            num_heads=nq, num_kv_heads=nkv, num_index_heads=niq, eps=EPS,
        )  # fmt: skip


def test_streaming_q_store_scope():
    """Only the four-pair main kernel uses streaming stores."""
    for pairs, want in [(4, True), (1, False)]:
        exe = host._get_kernel(*TP4, ROT, pairs)
        assert exe.kernel_name.endswith("_qnt") == want


@pytest.mark.parametrize("case", list(PREFILL_CASES))
@torch.inference_mode()
def test_tp4_prefill_variant_bytes(case, monkeypatch):
    """TP4 heads at prefill sizes: the variant the entry runs from PREFILL_TOKENS tokens meets
    the contract and writes the main kernel's bytes (q, index-q, every page byte; stale
    pages random, ~1% of the index slots unmapped, 37 padding tokens with slot -1)."""
    require_gfx950()
    batch = Batch(*PREFILL_CASES[case], TP4, (0.37, 0.21), seed=2, index_skip=0.01)
    got = batch.run()
    check(batch, got, batch.reference(), f"tp4 prefill variant {case}")
    monkeypatch.setattr(host, "PREFILL_TOKENS", 1 << 30)  # the main kernel at this size
    assert_bitwise(got, batch.run(), case)


@pytest.mark.parametrize("m", [1536, 1537, 1538, 2048, 3071, 4095, 6144])
@torch.inference_mode()
def test_tp4_q16_prefill_window_bytes(m, monkeypatch):
    """The prefill window from ``PREFILL_TOKENS`` (four pairs, streaming q
    stores): its edges, odd rows, a zero row, stale pages and disabled tail slots meet the
    contract and write the four-pair main kernel's bytes."""
    require_gfx950()
    batch = Batch([125], [m - 37], 37, TP4, (0.37, 0.21), seed=m, index_skip=0.01)
    batch.qkv[0].zero_()
    got = batch.run()
    check(batch, got, batch.reference(), f"q16 prefill window {m}")
    monkeypatch.setattr(host, "PREFILL_TOKENS", 1 << 30)
    assert_bitwise(got, batch.run(), f"q16 prefill window {m}")


@pytest.mark.parametrize("m", [24575, 24576, 24577, 28673, 32768, 32769])
@torch.inference_mode()
def test_tp4_streaming_prefill_bytes(m, monkeypatch):
    """Both window edges, a zero row, stale pages and disabled tail slots stay bitwise."""
    require_gfx950()
    batch = Batch([125], [m - 37], 37, TP4, (0.37, 0.21), seed=m, index_skip=0.01)
    batch.qkv[0].zero_()
    got = batch.run()
    check(batch, got, batch.reference(), f"streaming prefill {m}")
    monkeypatch.setattr(host, "PREFILL_TOKENS", 1 << 30)
    assert_bitwise(got, batch.run(), f"streaming prefill {m}")


def test_tp4_streaming_prefill_scope():
    for m in (24575, 24576, 32768, 32769):
        want = 8 if 24576 <= m <= 32768 else 4
        assert host._prefill_token_pairs(m) == want
    assert host._get_prefill_kernel(*TP4, ROT, 8).kernel_name.endswith("_qnt")
    assert host._get_prefill_kernel(*TP4, ROT, 4).kernel_name.endswith("_qnt")
    assert host.PREFILL_TOKENS == 1537
    assert host._main_token_pairs(488) == 1
    assert host._main_token_pairs(489) == 4
    assert host._main_token_pairs(511) == 4
    assert host._main_token_pairs(512) == 4
    assert host._prefill_token_pairs(512) == 4


@pytest.mark.parametrize("m", [489, 491, 500, 511, 512])
@torch.inference_mode()
def test_tp4_fourpair_boundary_bytes(m, monkeypatch):
    require_gfx950()
    batch = Batch([125], [m - 3], 3, TP4, (0.37, 0.21), seed=m, index_skip=0.01)
    batch.qkv[0].zero_()
    got = batch.run()
    check(batch, got, batch.reference(), f"four-pair boundary {m}")
    monkeypatch.setattr(host, "_main_token_pairs", lambda *args: 1)
    assert_bitwise(got, batch.run(), f"four-pair boundary {m}")


@pytest.mark.parametrize("m", [289, 303, 320, 351, 384])
@torch.inference_mode()
def test_tp4_streaming_onepair_bytes(m, monkeypatch):
    require_gfx950()
    batch = Batch([125], [m - 1], 1, TP4, (0.37, 0.21), seed=m, index_skip=0.01)
    batch.qkv[0].zero_()
    got = batch.run()
    check(batch, got, batch.reference(), f"streaming one-pair {m}")
    monkeypatch.setattr(host, "_get_streaming_q16_kernel",
                        lambda: host._get_kernel(*TP4, ROT, 1))
    assert_bitwise(got, batch.run(), f"streaming one-pair {m}")


# (tokens, split count): the QKV GEMM's deferred rows (split 3 at 1-128 tokens, 2 at 193-288) and other split counts
REDUCE_CASES = [(1, 3), (7, 2), (8, 3), (16, 3), (32, 4), (64, 4), (68, 4), (132, 2), (172, 8),
                (256, 3), (600, 3)]


@pytest.mark.parametrize("m,split_k", REDUCE_CASES)
@torch.inference_mode()
def test_splitk_reduce_contract(m, split_k):
    """The split-K reduce kernel, the last token graph padding."""
    require_gfx950()
    batch = Batch([3], [m - 1], 1, TP4, (0.5, 2.0), seed=m)
    slab = make_slab(batch, split_k, m)
    check(batch, batch.run_reduce(slab), batch.reference(slab=slab), f"reduce m{m} s{split_k}")


@pytest.mark.parametrize("iq_mode", ["rank", "cols"])
@pytest.mark.parametrize("case", ["dspark", "prefill600"])
@torch.inference_mode()
def test_index_q_out_view(iq_mode, case):
    """index_q_out as a slice of a bigger buffer (the context-parallel indexer's
    [4, n, 128] buffer, or a column slice); bytes outside it stay untouched."""
    require_gfx950()
    starts, lens = DSPARK if case == "dspark" else ([7], [600])
    batch = Batch(starts, lens, 5, TP4)
    n, w = batch.n, batch.niq * D
    if iq_mode == "rank":
        buf = torch.full((4, n, w), 0x7F, dtype=torch.uint8, device="cuda")
        iq = buf[2]
    else:
        buf = torch.full((n, 3 * w), 0x7F, dtype=torch.uint8, device="cuda")
        iq = buf[:, w : 2 * w]
    outs = batch.outputs()
    outs = (outs[0], iq.view(torch.float8_e4m3fn), outs[2], outs[3])
    check(batch, batch.run(outs=outs), batch.reference(), f"{iq_mode} {case}")
    iq.fill_(0x7F)
    assert bool((buf == 0x7F).all()), "bytes outside index_q_out written"


@torch.inference_mode()
def test_64k_tokens():
    require_gfx950()
    batch = Batch([0, 777], [32768, 32768], 0, TP4)
    check(batch, batch.run(), batch.reference(), "64K")


@pytest.mark.parametrize("heads", list(HEADS))
@torch.inference_mode()
def test_all_padding(heads):
    """A graph-capture dummy batch: every slot -1, the pages stay as they were."""
    require_gfx950()
    batch = Batch([], [], 64, HEADS[heads])
    got = batch.run()
    check(batch, got, batch.reference(), "all padding")
    assert torch.equal(got[2], batch.kv_full) and torch.equal(got[3], batch.idx_full)


@pytest.mark.parametrize("variant", ["main", "prefill"])
@torch.inference_mode()
def test_block_outer_pages(variant, monkeypatch):
    """Each block holds the layers' main and index pages in layer order ([M0 I0 M1 I1]):
    writing layer 0 through its strided views stores the bytes of dense caches and
    leaves layer 1's pages untouched."""
    require_gfx950()
    if variant == "main":
        monkeypatch.setattr(host, "PREFILL_TOKENS", 1 << 30)  # the main kernel at this size
    batch = Batch(*PREFILL_CASES["n4097_req3"], TP4, (0.37, 0.21), seed=1, index_skip=0.01)
    nb = batch.kv_full.shape[0]
    kv_page, idx_page = batch.main, BLK * WIDTH
    block = torch.randint(0, 256, (nb, 2 * (kv_page + idx_page)), dtype=torch.uint8, device="cuda")
    untouched = block[:, kv_page + idx_page :].clone()
    wide_kv = block[:, :kv_page].view(nb, 2 * batch.nkv, BLK, WIDTH)
    wide_idx = block[:, kv_page : kv_page + idx_page].view(nb, BLK, WIDTH)
    dense_kv, dense_idx = wide_kv.clone(), wide_idx.clone()
    for kv, idx in ((dense_kv, dense_idx), (wide_kv, wide_idx)):
        q, iq, _, _ = batch.outputs()
        norm_rope_nvfp4_insert(
            batch.qkv, *batch.w, batch.cs, batch.pos, batch.slots, batch.islots, kv, idx,
            batch.ks, batch.vs, q, iq, **batch.kwargs(), prefill=variant == "prefill",
        )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(dense_kv, wide_kv)
    assert torch.equal(dense_idx, wide_idx)
    assert torch.equal(block[:, kv_page + idx_page :], untouched)


# --- graph replay and determinism -------------------------------------------------------

# (path, heads, case): every kernel path at a size it serves
PATHS = {
    "main_pairs1_q16": ("main", TP4, CASES["dspark172_pad20"]),
    "streaming_onepair_q16": ("main", TP4, ([125], [299], 1)),
    "main_pairs4_q16": ("main", TP4, ([0, 300], [500, 533], 0)),
    "prefill_pairs4_q16": ("prefill", TP4, PREFILL_CASES["n4097_req3"]),
    "prefill_pairs8_q16": ("prefill", TP4, ([125], [24540], 37)),
    "reduce_q16_pairs1_s3": ("reduce3", TP4, ([3], [31], 1)),
    "reduce_q16_pairs2_s3": ("reduce3", TP4, ([3], [127], 1)),
    "reduce_q16_pairs2_s2": ("reduce2", TP4, ([125], [150], 1)),
    "reduce_q16_pairs1_s2": ("reduce2", TP4, ([125, 254], [123, 132], 1)),
}


def path_runner(path, batch, slab):
    """A call of ``path`` on ``batch``'s inputs into given outputs."""
    if path.startswith("reduce"):
        return lambda outs: batch.run_reduce(slab, outs=outs)
    return lambda outs: batch.run(prefill=path.startswith("prefill"), outs=outs)


@pytest.mark.parametrize("name", list(PATHS))
@torch.inference_mode()
def test_graph_replay(name):
    """Captured once, replayed on new qkv / positions / slots / pages in place, the
    outputs refilled with NaN / 0x7F before the replay."""
    require_gfx950()
    path, heads, case = PATHS[name]
    a = Batch(*case, heads, (0.5, 2.0), seed=1, index_skip=0.01)
    b = Batch(*case, heads, (0.5, 2.0), seed=2, index_skip=0.01)
    b.w, b.cs = a.w, a.cs  # captured with the layer's weights and RoPE cache
    split_k = int(path[-1]) if path.startswith("reduce") else 0
    slab_a = make_slab(a, split_k, 1) if split_k else None
    slab_b = make_slab(b, split_k, 2) if split_k else None
    outs = a.outputs()
    qkv, slab = a.qkv.clone(), None if slab_a is None else slab_a.clone()
    pos, slots, islots = a.pos.clone(), a.slots.clone(), a.islots.clone()
    q, iq, kv_full, idx_full = outs
    nk = a.kwargs()

    def run():
        if slab is not None:
            splitk_reduce_norm_rope_nvfp4_insert(
                slab, split_k, *a.w, a.cs, pos, slots, islots, a.kv(kv_full), a.idx(idx_full),
                a.ks, a.vs, q, iq, **nk)  # fmt: skip
        else:
            norm_rope_nvfp4_insert(
                qkv, *a.w, a.cs, pos, slots, islots, a.kv(kv_full), a.idx(idx_full), a.ks, a.vs,
                q, iq, **nk, prefill=path.startswith("prefill"))  # fmt: skip

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    qkv.copy_(b.qkv)
    if slab is not None:
        slab.copy_(slab_b)
    pos.copy_(b.pos)
    slots.copy_(b.slots)
    islots.copy_(b.islots)
    kv_full.copy_(b.kv_full)
    idx_full.copy_(b.idx_full)
    q.fill_(float("nan"))
    iq.view(torch.uint8).fill_(0x7F)
    graph.replay()
    torch.cuda.synchronize()
    check(b, outs, b.reference(slab_b), f"replay {name}")


@pytest.mark.parametrize("name", list(PATHS))
@torch.inference_mode()
def test_deterministic(name):
    """Three launches from the same stale pages are equal bit for bit."""
    require_gfx950()
    path, heads, case = PATHS[name]
    batch = Batch(*case, heads, (0.5, 2.0), seed=3, index_skip=0.01)
    split_k = int(path[-1]) if path.startswith("reduce") else 0
    run = path_runner(path, batch, make_slab(batch, split_k, 3) if split_k else None)
    outs = [run(batch.outputs()) for _ in range(3)]
    torch.cuda.synchronize()
    for other in outs[1:]:
        assert_bitwise(outs[0], other, name)


# --- the variants write the main kernel's bytes -----------------------------------------


@pytest.mark.parametrize("m,split_k", REDUCE_CASES)
@torch.inference_mode()
def test_splitk_reduce_bitwise(m, split_k):
    """The reduce kernel equals the reduce (``((0 + s0) + s1) ...`` in fp32, bf16 store)
    followed by the main kernel, bit for bit."""
    require_gfx950()
    batch = Batch([3], [m - 1], 1, TP4, (0.5, 2.0), seed=m)
    slab = make_slab(batch, split_k, m)
    fused = batch.run_reduce(slab)
    batch.qkv = ref.splitk_reduce(slab)
    assert_bitwise(fused, batch.run(), f"m{m} s{split_k}")


# --- known limitation -------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="limitation (kernel.py, kernel_prefill_q64_kv4_iq4.py): a V wave writes the "
    "whole run of tokens in one octet from the run's first token and counts the run "
    "by slots in the octet, so an unmapped (-1) token inside a run is written at its "
    "position and the run's later tokens get a second, racing writer. vLLM pads only "
    "at the batch tail, so serving never hits it.",
)
@pytest.mark.parametrize("variant", ["main", "prefill"])
@torch.inference_mode()
def test_unmapped_token_inside_v_octet(variant, monkeypatch):
    require_gfx950()
    if variant == "main":
        monkeypatch.setattr(host, "PREFILL_TOKENS", 1 << 30)  # the main kernel at this size
    starts, lens, pad = PREFILL_CASES["n4097_req3"]
    batch = Batch(starts, lens, pad, TP4, (0.37, 0.21), seed=4)
    # tokens 2 and 9 of the first request: inside their octets' runs
    batch.slots[2] = -1
    batch.slots[9] = -1
    check(batch, batch.run(prefill=variant == "prefill"), batch.reference(), variant)
