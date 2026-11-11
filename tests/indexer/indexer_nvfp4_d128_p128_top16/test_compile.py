# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

The engine's warm-up on ROCm is the decode CUDA-graph capture, every captured batch size at ``seq_lens =
max_seq_len = max_model_len``; nothing warms the prefill indexer. A variant is the set of kernels a call launches
(their names carry every compile key); ``launched`` reads it from the host plan with the launches stubbed out. The
rank-owned kernels wait for their peers, so their checks read the host plan only (``test_split4_rank_owned`` runs
them).

Per served configuration: (1) the engine's warm-up, then the inputs an eager step brings (a mixed prefill + decode
step, or a batch above the captured sizes); a probe that compiles is a strict xfail. (2) A warm-up of the inputs of
the served range that bring a new kernel, then other probes: nothing compiles.
"""

import random

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, cdiv, score_width
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host as decode_host
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_decode
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import host as prefill_host
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import nvfp4_index_score_prefill
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import host as topk_host
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    index_topk_decode_indices,
    index_topk_prefill,
    index_topk_prefill_indices,
)

from .inputs import index_cache, queries, require_gfx950

INIT, LOCAL = 0, 1
MAX_MODEL_LEN = 1 << 20  # MiniMax-M3.1's max_position_embeddings, the served max_model_len

# (name, index heads, query rows, captured request counts, max requests)
DECODE = [
    # four GPUs split by heads, 4 rows, the split4 scorer off: graph sizes up to 1024 tokens = 256 requests
    ("h1q4", 1, 4, [1, 2, 4] + list(range(8, 257, 8)), 256),
    # the TP4 one-GPU route (one head per GPU), 8 rows: 1..64 requests
    ("h1q8", 1, 8, list(range(1, 65)), 64),
]
PREFILL_HEADS = [1]

PREFILL_WARMUP_GAP = pytest.mark.xfail(
    strict=True,
    reason=(
        "no ROCm warm-up reaches the prefill indexer: the scorer (shared-dequant / "
        "512-row tile) and the top-k (5 windows by max_seq_len) compile on the "
        "first prefill that needs them"
    ),
)


@pytest.fixture(autouse=True)
def fresh_compile_state():
    """Every test starts as a fresh server: earlier tests' launchers (and their compiled functions) are dropped."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host as split4_host
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import decode_indices_kernel

    getters = [
        topk_host._get_kernel,
        topk_host._get_prefill_indices_kernel,
        decode_indices_kernel._get_decode_indices_kernel,
        decode_host._get_kernel,
        decode_host._get_kernel_devpos,
        prefill_host._get_kernel,
        prefill_host._get_kernel_shared,
        prefill_host._get_kernel_causal,
        split4_host._score_push_rank_owned_kernel,
    ]
    for g in getters:
        g.cache_clear()
    yield


class CountCompiles:
    def __init__(self, monkeypatch):
        self.compiled = []
        real = flyc.compile
        outer = self

        class Counting:
            def __init__(self, compiler):
                self.compiler = compiler
            def __getitem__(self, hints):
                return Counting(self.compiler[hints])
            def __call__(self, fn, *args, **kwargs):
                outer.compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
                return self.compiler(fn, *args, **kwargs)

        monkeypatch.setattr(flyc, "compile", Counting(real))


def warm_kernels(call, inputs, monkeypatch):
    """Runs the inputs that launch a kernel not seen yet and returns the kernels; the warm-up must compile exactly
    the kernels the host plan names, each once (a flag the kernel ignores would compile one kernel twice)."""
    seen, warm = set(), []
    for inp in inputs:
        kernels = set(launched(lambda inp=inp: call(*inp)))
        if not kernels <= seen:
            seen |= kernels
            warm.append(inp)
    with pytest.MonkeyPatch.context() as mp:
        counter = CountCompiles(mp)
        for inp in warm:
            call(*inp)
        torch.cuda.synchronize()
    assert sorted(counter.compiled) == sorted(seen), (sorted(counter.compiled), sorted(seen))
    return seen


# max_seq_len over the served range: every page-table top-k window (8192 tokens
# each, 1..32) and the captured max_model_len
SERVED_MSL = [100, 3000] + [k * 8192 for k in range(1, 33)] + [MAX_MODEL_LEN]


def launched(call) -> tuple:
    """The kernels ``call`` launches, read from the host plan without launching them."""
    names = []

    def record(exe, *args, **kwargs):
        names.append(exe.kernel_name)

    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host as split4_host

    with pytest.MonkeyPatch.context() as mp:
        for mod in (topk_host, decode_host, prefill_host, split4_host):
            mp.setattr(mod, "_run_compiled", record)
        call()
    return tuple(names)


# ---------------------------------------------------------------- decode


class Decode:
    """Buffers for the decode chain up to ``max_reqs``: the scorer, then the indices top-k."""

    def __init__(self, heads, ql, max_reqs):
        self.heads, self.ql = heads, ql
        rows = max_reqs * ql
        self.cache = index_cache(64)
        self.q = queries(rows, heads)
        self.score = torch.empty(heads, rows, score_width(MAX_MODEL_LEN), device="cuda")
        self.topk = torch.empty(heads, rows, 16, dtype=torch.int32, device="cuda")

    def call(self, lens, max_seq_len):
        n = len(lens)
        rows = n * self.ql
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        table = torch.randint(0, 64, (n, cdiv(max(max(lens), 1), BLK)), dtype=torch.int32, device="cuda")
        score, topk = self.score[:, :rows], self.topk[:, :rows]
        nvfp4_index_score_decode(
            self.q[:rows], self.cache, score, table, seq, INIT, LOCAL, self.ql, max_seq_len
        )
        index_topk_decode_indices(score, topk, seq, max_seq_len=max_seq_len, query_len=self.ql)

    def capture_warmup(self, captured):
        for n in captured:
            self.call([MAX_MODEL_LEN] * n, MAX_MODEL_LEN)
        torch.cuda.synchronize()


def eager_steps(max_reqs, seed):
    """Eager decode steps: random request counts and lengths, the step's own max_seq_len."""
    rng = random.Random(seed)
    steps = []
    for hi in (3000, 40000, 130000, 300000, MAX_MODEL_LEN):
        n = rng.randint(1, max_reqs)
        lens = [rng.randint(1, hi) for _ in range(n)]
        steps.append((lens, max(lens)))
    return steps


@pytest.mark.parametrize(
    "name,heads,ql,captured,max_reqs", DECODE, ids=[c[0] for c in DECODE]
)
@torch.inference_mode()
def test_decode_no_compile_after_capture(name, heads, ql, captured, max_reqs, monkeypatch):
    require_gfx950()
    chain = Decode(heads, ql, max_reqs)
    chain.capture_warmup(captured)
    counter = CountCompiles(monkeypatch)
    for lens, msl in eager_steps(max_reqs, 1):
        chain.call(lens, msl)
        assert not counter.compiled, (name, len(lens), msl, counter.compiled)


@pytest.mark.parametrize(
    "name,heads,ql,captured,max_reqs", DECODE, ids=[c[0] for c in DECODE]
)
@torch.inference_mode()
def test_decode_no_compile_after_kernel_warmup(name, heads, ql, captured, max_reqs, monkeypatch):
    require_gfx950()
    chain = Decode(heads, ql, max_reqs)
    inputs = [([msl] * n, msl) for n in captured for msl in SERVED_MSL]
    kernels = warm_kernels(chain.call, inputs, monkeypatch)
    print(f"decode {name}: {sorted(kernels)}")
    counter = CountCompiles(monkeypatch)
    for lens, msl in eager_steps(max_reqs, 4):
        chain.call(lens, msl)
        assert not counter.compiled, (name, len(lens), msl, counter.compiled)


# ---------------------------------------------------------------- prefill


class Prefill:
    """The served prefill chain on ragged rows: the scorer, then the indices top-k (or the page-table top-k of the
    FP8 main-KV attend)."""

    def __init__(self, heads, table_topk=False):
        self.heads, self.table_topk = heads, table_topk
        self.cache = index_cache(64)

    def call(self, q_lens, seq_lens):
        heads = self.heads
        total = sum(q_lens)
        msl = max(seq_lens)
        q = queries(total, heads)
        seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
        table = torch.randint(0, 64, (len(q_lens), cdiv(msl, BLK)), dtype=torch.int32, device="cuda")
        score = torch.empty(heads, total, score_width(msl), device="cuda")
        topk = torch.empty(heads, total, 16, dtype=torch.int32, device="cuda")
        pos = torch.cat([torch.arange(s - n, s, device="cuda") for n, s in zip(q_lens, seq_lens)])
        nvp = (pos // BLK + 1).to(torch.int32)
        nvfp4_index_score_prefill(q, self.cache, score, table, cu, seq, INIT, LOCAL, max(q_lens), msl)
        if self.table_topk:
            rid = torch.repeat_interleave(
                torch.arange(len(q_lens), device="cuda"), torch.tensor(q_lens, device="cuda")
            ).to(torch.int32)
            index_topk_prefill(
                score, topk, table, seq,
                torch.empty(total * heads, 128, dtype=torch.int32, device="cuda"),
                torch.empty(total * heads, dtype=torch.int32, device="cuda"),
                max_seq_len=msl, block_size=BLK, num_valid_pages=nvp, row_req_id=rid,
                kv_lens=(pos + 1).to(torch.int32), num_kv_heads=heads, pages_per_block=8,
            )  # fmt: skip
        else:
            index_topk_prefill_indices(score, topk, max_seq_len=msl, block_size=BLK, num_valid_pages=nvp)


# (q_lens, seq_lens): one request or a ragged step, contexts across the top-k windows and both scorer kernels
PREFILL_GRID = [([n], [max(n, ctx)]) for n in (64, 512, 2048) for ctx in SERVED_MSL[2:]] + [
    ([1000, 64, 2048], [1000, 60000, 2048]),
    ([3072, 3072], [203072, 3072]),
    ([4096, 4096, 4096, 4096], [4096, 50000, 120000, 160000]),
]
# probes: other steps of the served range
PREFILL_PROBES = [
    ([300], [300]),
    ([4096], [4096]),
    ([700, 1500], [30000, 1500]),
    ([8192], [65536]),
    ([2000, 513, 3000], [9000, 513, 20000]),
    ([1024], [250000]),
    ([16384], [131072]),
    ([128], [1 << 20]),
]


@pytest.mark.parametrize(
    "heads", [pytest.param(h, marks=PREFILL_WARMUP_GAP, id=f"h{h}") for h in PREFILL_HEADS]
)
@torch.inference_mode()
def test_prefill_no_compile_after_engine_warmup(heads, monkeypatch):
    """After the engine's warm-up (decode capture only), a first prefill."""
    require_gfx950()
    Decode(heads, 4, 8).capture_warmup([1, 8])
    chain = Prefill(heads)
    counter = CountCompiles(monkeypatch)
    for q_lens, seq_lens in PREFILL_PROBES:
        chain.call(q_lens, seq_lens)
        assert not counter.compiled, (heads, q_lens, seq_lens, counter.compiled)


@pytest.mark.parametrize("table_topk", [False, True], ids=["indices", "table"])
@pytest.mark.parametrize("heads", PREFILL_HEADS, ids=[f"h{h}" for h in PREFILL_HEADS])
@torch.inference_mode()
def test_prefill_no_compile_after_kernel_warmup(heads, table_topk, monkeypatch):
    require_gfx950()
    chain = Prefill(heads, table_topk)
    kernels = warm_kernels(chain.call, PREFILL_GRID, monkeypatch)
    print(f"prefill h{heads} {'table' if table_topk else 'indices'}: {sorted(kernels)}")
    counter = CountCompiles(monkeypatch)
    for q_lens, seq_lens in PREFILL_PROBES:
        chain.call(q_lens, seq_lens)
        assert not counter.compiled, (heads, q_lens, seq_lens, counter.compiled)


@torch.inference_mode()
def test_variants_are_few():
    """The served ranges reach a small set of kernels (printed for the README)."""
    require_gfx950()
    for name, heads, ql, captured, max_reqs in DECODE:
        chain = Decode(heads, ql, max_reqs)
        kernels = set()
        for n in range(1, max_reqs + 1):
            for msl in SERVED_MSL:
                kernels.update(launched(lambda: chain.call([msl] * n, msl)))
        print(f"decode {name}: {len(kernels)}")
        assert len(kernels) <= 4
    for heads in PREFILL_HEADS:
        for table_topk in (False, True):
            chain = Prefill(heads, table_topk)
            kernels = set()
            for shape in PREFILL_GRID:
                kernels.update(launched(lambda s=shape: chain.call(*s)))
            print(f"prefill h{heads} {'table' if table_topk else 'indices'}: {len(kernels)}")
            assert len(kernels) <= (34 if table_topk else 7)


# ---------------------------------------------------------------- rank-owned index cache

# 4 / 6 / 8 rows per request, up to 256 requests
RANK_OWNED_QL = [4, 6, 8]
RANK_OWNED_CAPTURED = [1, 2, 4] + list(range(8, 257, 8))


class RankOwned:
    """Rank ``rank``'s decode chain on the rank-owned index cache, built without IPC for the host plan only:
    ``IndexSplit4RankOwned.score_push`` + the rank-owned-order indices top-k (with ``fused`` ``score_topk``)."""

    def __init__(self, ql, rank=0, max_reqs=256, fused=False):
        self.fused = fused
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host as split4_host

        self.ql, self.rank = ql, rank
        width = score_width(MAX_MODEL_LEN)
        dec = split4_host.IndexSplit4RankOwned.__new__(split4_host.IndexSplit4RankOwned)
        dec.rank, dec.query_len = rank, ql
        dec.max_rows, dec.max_width = max_reqs * ql, width
        dec.score_ptrs = dec.flag_ptrs = dec.iq_ptrs = dec.mark_ptrs = [0] * 4
        dec.score_rcv = torch.empty(max_reqs * ql * width, device="cuda")
        dec.counts = torch.zeros(64, dtype=torch.int32, device="cuda")
        self.dec = dec
        self.cache = torch.empty(16, BLK, 72, dtype=torch.uint8, device="cuda")
        self.q = queries(max_reqs * ql, 1)[:, 0]

    def decode(self, lens, max_seq_len):
        n = len(lens)
        rows = n * self.ql
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        table = torch.zeros(n, cdiv(max(max(lens), 1), BLK), dtype=torch.int32, device="cuda")
        width = score_width(max_seq_len)
        topk = torch.empty(1, rows, 16, dtype=torch.int32, device="cuda")
        if self.fused:
            self.dec.score_topk(
                self.q[:rows], self.cache, table, seq, INIT, LOCAL, width, topk, max_seq_len
            )
            return
        score = self.dec.score_push(self.q[:rows], self.cache, table, seq, INIT, LOCAL, width)
        index_topk_decode_indices(
            score, topk, seq, max_seq_len=max_seq_len, query_len=self.ql,
            rank_owned_order=True,
        )  # fmt: skip

    def capture_kernels(self):
        """What the engine's warm-up (decode graph capture at max_model_len) runs."""
        kernels = set()
        for n in RANK_OWNED_CAPTURED:
            kernels.update(launched(lambda n=n: self.decode([MAX_MODEL_LEN] * n, MAX_MODEL_LEN)))
        return kernels


def _assert_covered(warm, probes, what):
    for name, call in probes:
        missing = set(launched(call)) - warm
        assert not missing, (what, name, sorted(missing))


def _decode_probes(chain, seed):
    return [
        (f"n{len(lens)} msl{msl}", lambda lens=lens, msl=msl: chain.decode(lens, msl))
        for lens, msl in eager_steps(256, seed)
    ]


@pytest.mark.parametrize("ql", RANK_OWNED_QL, ids=[f"q{q}" for q in RANK_OWNED_QL])
@torch.inference_mode()
def test_rank_owned_decode_no_compile_after_capture(ql):
    """The served decode (indices top-k): the capture covers the eager steps."""
    require_gfx950()
    chain = RankOwned(ql)
    _assert_covered(chain.capture_kernels(), _decode_probes(chain, 5), ql)


@pytest.mark.parametrize("ql", RANK_OWNED_QL, ids=[f"q{q}" for q in RANK_OWNED_QL])
@torch.inference_mode()
def test_rank_owned_decode_fused_no_compile_after_capture(ql):
    """``score_topk`` (one launch at <= 128 rows, two above): the capture covers the eager steps."""
    require_gfx950()
    chain = RankOwned(ql, fused=True)
    _assert_covered(chain.capture_kernels(), _decode_probes(chain, 5), ql)


@pytest.mark.parametrize("ql", RANK_OWNED_QL, ids=[f"q{q}" for q in RANK_OWNED_QL])
@torch.inference_mode()
def test_rank_owned_decode_fused_no_compile_after_kernel_warmup(ql):
    """``score_topk``: a warm-up over every top-k window at 1, 32, 64 and 256 requests (fused, four-wave and
    one-wave paths) covers the eager decode steps."""
    require_gfx950()
    chain = RankOwned(ql, fused=True)
    warm = set()
    for msl in SERVED_MSL:
        for n in (1, 32, 64, 256):
            warm.update(launched(lambda msl=msl, n=n: chain.decode([msl] * n, msl)))
    assert len(warm) == (5 if ql > 4 else 4), sorted(warm)
    _assert_covered(warm, _decode_probes(chain, 6), ql)


@pytest.mark.parametrize("ql", RANK_OWNED_QL, ids=[f"q{q}" for q in RANK_OWNED_QL])
@torch.inference_mode()
def test_rank_owned_decode_no_compile_after_kernel_warmup(ql):
    """A warm-up over every top-k window covers the eager decode steps."""
    require_gfx950()
    chain = RankOwned(ql)
    warm = set()
    for msl in SERVED_MSL:
        for n in (8, 256):
            warm.update(launched(lambda msl=msl, n=n: chain.decode([msl] * n, msl)))
    print(f"rank-owned decode q{ql}: {len(warm)} kernels")
    assert len(warm) == 4, sorted(warm)
    _assert_covered(warm, _decode_probes(chain, 6), ql)
