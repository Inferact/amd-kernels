# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The shared expert's second-stream tier (96..30720 rows): one side stream per device, every buffer from the caller's
allocator pool, the DBO opt-out (``set_dbo_query``). Two caller streams issue the entry in turn without a host sync,
ordered on the GPU as DBO's microbatches are (two calls must not run at once on one device: GEMM12's ready counters are
per device); outputs equal the serial shared expert bit for bit, reserved memory grows by at most two [m, 6144] bf16
buffers after the warm rounds and then stays flat, the allocator never retries and no segment belongs to the side
stream."""

import importlib

import pytest
import torch

from .workload import make_activation, make_experts, require_gfx950

sh = importlib.import_module("amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert")


@pytest.fixture(scope="module")
def complete_weights():
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights

    require_gfx950()
    g = torch.Generator(device="cuda").manual_seed(607)

    def weight(n, k):
        q = (torch.randn((n, k), device="cuda", generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
        s = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
        return q, s

    shared = SharedWeights.from_row_major(*weight(1536, 6144), *weight(6144, 768))
    routed, _ = make_experts()
    gate = torch.randn((128, 6144), generator=g, device="cuda") / 6144**0.5
    bias = torch.randn((128,), generator=g, device="cuda") * 0.01
    return MoeWeights.from_row_major(gate, bias, *routed, shared=shared)


def _side_segments(stream):
    return [s for s in torch.cuda.memory_snapshot() if s.get("stream") == stream.cuda_stream]


def test_one_side_stream_per_device(complete_weights):
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import SharedWeights

    w = complete_weights.shared_weights
    other = SharedWeights.from_row_major(
        torch.zeros((1536, 6144), dtype=torch.uint8, device="cuda"),
        torch.zeros((1536, 192), dtype=torch.uint8, device="cuda"),
        torch.zeros((6144, 768), dtype=torch.uint8, device="cuda"),
        torch.zeros((6144, 24), dtype=torch.uint8, device="cuda"))
    assert other.stream is w.stream is sh.side_stream(torch.device("cuda"))


@pytest.mark.parametrize("m", [4096, 11792, 16384, 30720])
def test_two_caller_streams_no_side_pool(complete_weights, m, monkeypatch):
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe

    w = complete_weights
    xs = [make_activation(m, seed=700 + m + i) for i in range(2)]
    assert sh.shared_parallel_for(m) and not sh.shared_in_persistent_for(m)
    with monkeypatch.context() as mp:  # the serial shared expert: the reference
        mp.setattr(sh, "shared_parallel_for", lambda n: False)
        ref = [mxfp4_moe(x, w, return_partials=True) for x in xs]
        ref_reduced = [mxfp4_moe(x, w) for x in xs]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # the pools hold only this test's blocks (earlier tests' cached sizes split differently)
    side = w.shared_weights.stream
    callers = [torch.cuda.Stream(), torch.cuda.Stream()]

    def round_():
        outs = []
        prev = torch.cuda.current_stream()
        for x, s in zip(xs, callers):  # microbatch 0 then 1, no host sync in between
            s.wait_stream(prev)
            with torch.cuda.stream(s):
                outs.append((mxfp4_moe(x, w, return_partials=True), mxfp4_moe(x, w)))
            prev = s
        torch.cuda.current_stream().wait_stream(prev)
        return outs

    def check(outs):
        for (parts, reduced), (rp, rs), rr in zip(outs, ref, ref_reduced):
            assert torch.equal(parts[0], rp) and torch.equal(parts[1], rs)
            assert torch.equal(reduced, rr)

    for _ in range(3):  # warm rounds fill the callers' pools (blocks the side stream still reads wait a round)
        check(round_())
    torch.cuda.synchronize()
    stats0 = torch.cuda.memory_stats()
    # The blocks the side stream read are freed with record_stream, so when they come back depends on host timing:
    # a busy host can split one more [m, 6144] bf16 segment in a late round, then the pools stay flat. A side pool per
    # layer (the bug this guards) grew GiBs per step. Bound the growth to two such buffers and require it to stop.
    slack = 2 * m * 6144 * 2
    for _ in range(6):  # each round's outputs are dropped before the next, as a server's step
        check(round_())
    torch.cuda.synchronize()
    stats1 = torch.cuda.memory_stats()
    for _ in range(12):
        check(round_())
    torch.cuda.synchronize()
    stats2 = torch.cuda.memory_stats()
    reserved = [st["reserved_bytes.all.current"] for st in (stats0, stats1, stats2)]
    assert stats2["num_alloc_retries"] == stats0["num_alloc_retries"]
    assert reserved[2] - reserved[0] <= slack, reserved
    assert reserved[2] - reserved[1] <= slack // 2, reserved  # no steady growth
    assert not _side_segments(side), "the allocator cached blocks under the side stream"


@pytest.mark.parametrize("m", [4096, 16384])
def test_dbo_query_keeps_caller_stream(complete_weights, m, monkeypatch):
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe, set_dbo_query

    w = complete_weights
    x = make_activation(m, seed=900 + m)
    expect = mxfp4_moe(x, w, return_partials=True)
    calls = []
    set_dbo_query(lambda: calls.append(1) or True)
    try:
        assert not sh.shared_parallel_for(m)
        caller = torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        with monkeypatch.context() as mp:  # the fork must not run: fail if it is called
            mp.setattr(sh, "shared_expert_parallel", lambda *a, **k: pytest.fail("forked inside DBO"))
            with torch.cuda.stream(caller):
                got = mxfp4_moe(x, w, return_partials=True)
        torch.cuda.current_stream().wait_stream(caller)
        torch.cuda.synchronize()
        assert calls
        assert torch.equal(got[0], expect[0]) and torch.equal(got[1], expect[1])
    finally:
        set_dbo_query(None)
    assert sh.shared_parallel_for(m) and not sh.DBO
