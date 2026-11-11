# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

``key`` mirrors the getter and key the host asks for per call (``_get_kernel``, ``_get_prefill_kernel``,
``_get_streaming_q16_kernel``, ``_get_reduce_kernel``); every test clears those caches, spies on the keys and counts
``flydsl.compiler.compile`` calls.

1. ``test_token_count_not_in_key``: per variant, warm one token count, probe six others of its range.
2. TP4: the warm-up is what CUDA graph capture launches (1 .. 1024 tokens), eager steps up to 65536 (DBO
   micro-batches of >= 4096 above 8192); covered with the init call ``warmup_prefill_kernels``, not without it (strict
   xfail).
"""

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import host

from .test_norm_rope_insert import EPS, ROT, TP4, require_gfx950

PROBES = 6
GETTERS = ("_get_kernel", "_get_prefill_kernel", "_get_reduce_kernel", "_get_streaming_q16_kernel")


def pairs(n):
    return 1 if n <= 512 else 4


def reduce_pairs(n, split):
    if split == 2 and 176 < n <= 288:
        return 1
    return 1 if n <= 32 else (2 if n <= 512 else 4)


def key(n, prefill=False, split=0):
    """The getter and key the host should ask for."""
    if split:
        return "_get_reduce_kernel", (*TP4, ROT, reduce_pairs(n, split), split)
    base_pairs = 4 if 489 <= n <= 512 else pairs(n)
    if (prefill or n >= host.PREFILL_TOKENS) and base_pairs % 2 == 0:
        tp = 8 if 24576 <= n <= 32768 else base_pairs
        return "_get_prefill_kernel", (*TP4, ROT, tp)
    if 289 <= n <= 384:
        return "_get_streaming_q16_kernel", ()
    return "_get_kernel", (*TP4, ROT, base_pairs)


class Data:
    """One batch of the largest size (a single request in contiguous slots); calls run
    on its first n tokens."""

    def __init__(self, n_max, splits=()):
        nq, nkv, niq = self.heads = TP4
        dev = "cuda"
        row = (nq + 2 * nkv + niq + 1) * 128
        self.qkv = torch.randn(n_max, row, device=dev).to(torch.bfloat16)
        self.slabs = {s: torch.randn(s, min(n_max, 1024), row, device=dev) for s in splits}
        self.w = [(torch.randn(128, device=dev) * 0.1).to(torch.bfloat16) for _ in range(4)]
        self.cs = (torch.rand(n_max + 8, ROT, device=dev) * 2 - 1).to(torch.bfloat16)
        self.pos = torch.arange(n_max, device=dev)
        pages = n_max // 128 + 2
        self.slots = torch.arange(n_max, device=dev) + 5
        self.kv = torch.zeros(pages, 2 * nkv, 128, 72, dtype=torch.uint8, device=dev)
        self.idx = torch.zeros(pages, 128, 72, dtype=torch.uint8, device=dev)
        self.q = torch.empty(n_max, nq * 128, dtype=torch.bfloat16, device=dev)
        self.iq = torch.empty(n_max, niq * 128, dtype=torch.float8_e4m3fn, device=dev)
        self.ks = torch.tensor(0.5, device=dev)
        self.vs = torch.tensor(2.0, device=dev)

    def call(self, n, prefill=False, split=0):
        nq, nkv, niq = self.heads
        args = (*self.w, self.cs, self.pos[:n], self.slots[:n], self.slots[:n], self.kv, self.idx,
                self.ks, self.vs, self.q[:n], self.iq[:n])  # fmt: skip
        kw = dict(num_heads=nq, num_kv_heads=nkv, num_index_heads=niq, eps=EPS)
        if split:
            slab = self.slabs[split][:, :n].contiguous()
            host.splitk_reduce_norm_rope_nvfp4_insert(slab, split, *args, **kw)
        else:
            host.norm_rope_nvfp4_insert(self.qkv[:n], *args, **kw, prefill=prefill)


class Spy:
    """Clears the host's kernel caches, records the keys asked for and the compiles."""

    def __init__(self, monkeypatch):
        self.keys, self.compiled = [], []
        for name in GETTERS:
            real = getattr(host, name)
            real.cache_clear()

            def asked(*args, _real=real, _name=name):
                self.keys.append((_name, args))
                return _real(*args)

            monkeypatch.setattr(host, name, asked)
        real_compile = flyc.compile

        def counting(fn, *args, **kwargs):
            self.compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
            return real_compile(fn, *args, **kwargs)

        monkeypatch.setattr(flyc, "compile", counting)

    def call(self, data, n, want, **kw):
        """One launch; checks the key the host asked for against ``want``."""
        self.keys.clear()
        data.call(n, **kw)
        assert self.keys == [want], f"n={n}: the host asked {self.keys}, the mirror says {want}"


# (prefill, split) -> token range of the variant set
RANGES = {
    "q16_kv1_iq1": (False, 0, range(1, 65537)),
    # the QKV's deferred rows (split 3 up to 128 tokens, split 2 at 129-288)
    "q16_kv1_iq1_split2": (False, 2, range(1, 513)),
    "q16_kv1_iq1_split3": (False, 3, range(1, 513)),
}


def warm_and_probe(served, variant):
    warm = {}
    for n in served:
        warm.setdefault(variant(n), n)
    rest = [n for n in served if n not in warm.values()]
    return warm, rest[:: max(1, len(rest) // PROBES)][:PROBES]


@pytest.mark.parametrize("name", list(RANGES))
@torch.inference_mode()
def test_token_count_not_in_key(name, monkeypatch):
    require_gfx950()
    prefill, split, served = RANGES[name]
    kw = dict(prefill=prefill, split=split)
    data = Data(served[-1], (split,) if split else ())
    spy = Spy(monkeypatch)
    warm, probe = warm_and_probe(served, lambda n: key(n, **kw))
    for n in warm.values():
        spy.call(data, n, key(n, **kw), **kw)
    torch.cuda.synchronize()
    spy.compiled.clear()
    for n in probe:
        spy.call(data, n, key(n, **kw), **kw)
        assert not spy.compiled, f"{name}: n={n} compiled {spy.compiled} after warm-up on {sorted(warm.values())}"


def check_tp4(monkeypatch, warm_sizes, served, n_max, warmup=False):
    data = Data(n_max)
    spy = Spy(monkeypatch)
    if warmup:
        host.warmup_prefill_kernels(*TP4, ROT)
    for n in warm_sizes:
        spy.call(data, n, key(n))
    torch.cuda.synchronize()
    spy.compiled.clear()
    late = {}
    for n in served:
        spy.call(data, n, key(n))
        if spy.compiled:
            late[n] = list(spy.compiled)
            spy.compiled.clear()
    print(f"tp4: late compiles {late}")
    assert not late, f"tp4: compiled after warm-up: {late}"


# vLLM's default capture sizes up to 1024
TP4_CAPTURE = [1, 2, 4, *range(8, 256, 8), *range(256, 1025, 16)]
TP4_EAGER = [1, 3, 7, 100, 333, 512, 513, 777, 1024, 1500, 4096, 8191, 12345, 32768, 65536]


@torch.inference_mode()
def test_tp4_capture_and_warmup_cover_serving(monkeypatch):
    """TP4 with the init call ``warmup_prefill_kernels(16, 1, 1)`` (the steps from
    ``PREFILL_TOKENS`` tokens run the prefill variant, which no capture size reaches)."""
    require_gfx950()
    # above 8192 tokens DBO runs two micro-batches of at least 4096 tokens each
    served = [n if n < 8192 else n // 2 for n in TP4_EAGER]
    check_tp4(monkeypatch, TP4_CAPTURE, served, 65536, warmup=True)


@pytest.mark.xfail(
    strict=True,
    reason="vLLM serving warm-up gap until the TP4 model calls warmup_prefill_kernels at init: "
    "the capture sizes (<= 1024 tokens) reach only the main kernel, so the first step of "
    ">= 1537 tokens compiles the prefill variant on the forward path",
)
@torch.inference_mode()
def test_tp4_capture_alone_covers_serving(monkeypatch):
    require_gfx950()
    served = [n if n < 8192 else n // 2 for n in TP4_EAGER]
    check_tp4(monkeypatch, TP4_CAPTURE, served, 65536)


def test_variants_are_few():
    """The served ranges reach a small set of variants (printed for the README)."""
    variants = sorted({key(n, *RANGES[k][:2]) for k in RANGES for n in RANGES[k][2]})
    print(f"{len(variants)} variants: {variants}")
    # main 1 / 4 pairs, the streaming one-pair tile, prefill 4 / 8 pairs; the deferred QKV rows 1 / 2 token pairs x
    # split 2 / 3
    assert len(variants) == 9


@torch.inference_mode()
def test_prefill_init_warms_both_pair_counts(monkeypatch):
    """One init call covers both prefill pair counts."""
    require_gfx950()
    spy = Spy(monkeypatch)
    host.warmup_prefill_kernels(*TP4)
    data = Data(32768)
    spy.compiled.clear()
    for n in (8192, 24576, 32768):
        data.call(n)
    torch.cuda.synchronize()
    assert not spy.compiled, spy.compiled
