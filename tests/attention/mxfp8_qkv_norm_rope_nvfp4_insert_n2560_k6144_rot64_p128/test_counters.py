# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host-thread counter reservation: distinct streams/captures must never alias a workspace."""

import concurrent.futures
import threading
import time

import pytest
import torch

from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import host


def _isolate(monkeypatch, *, shared_key):
    device = torch.device("cpu")
    monkeypatch.setattr(host, "_POOL", {})
    monkeypatch.setattr(host, "_COUNTERS", {})
    monkeypatch.setattr(host, "_COUNTER_SETS", 16)
    monkeypatch.setattr(host, "_counter_key", lambda d: (d, ("thread", 0 if shared_key else threading.get_ident())))
    return device


def _parallel_reserve(device):
    start = threading.Barrier(8)

    def reserve(_):
        start.wait(timeout=10)
        return host._counters(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(reserve, range(8)))


@pytest.mark.parametrize("shared_key", [False, True])
def test_counter_slices_reserved_atomically(monkeypatch, shared_key):
    """Yield the GIL during tensor slicing, as real torch tensor operations can do."""
    device = _isolate(monkeypatch, shared_key=shared_key)
    pool = torch.zeros(host._COUNTER_SETS * host._COUNTER_WORDS, dtype=torch.int32)
    slices = []

    class SlowPool:
        def __getitem__(self, section):
            time.sleep(0.02)
            slices.append(section)
            return pool[section]

    host._POOL[device] = SlowPool()
    results = _parallel_reserve(device)
    pointers = [t.data_ptr() for t in results]
    assert len(set(pointers)) == (1 if shared_key else 8)
    assert len(slices) == (1 if shared_key else 8)
    assert len(host._COUNTERS) == len(slices)
    if not shared_key:
        assert sorted(s.start for s in slices) == [i * host._COUNTER_WORDS for i in range(8)]
    for tensor in results:
        assert tensor.numel() == host._COUNTER_WORDS and not tensor.count_nonzero()


@pytest.mark.parametrize("shared_key", [False, True])
def test_counter_pool_initialized_once(monkeypatch, shared_key):
    """Concurrent first calls share one fully initialized pool; no unacknowledged slice is published."""
    device = _isolate(monkeypatch, shared_key=shared_key)
    real_zeros = torch.zeros
    allocations, acknowledgements = [], []

    def slow_zeros(*args, **kwargs):
        time.sleep(0.02)
        allocations.append(1)
        return real_zeros(*args, **kwargs)

    class Ready:
        def record(self, stream):
            pass

        def synchronize(self):
            time.sleep(0.02)
            assert not host._COUNTERS
            acknowledgements.append(1)

    monkeypatch.setattr(torch, "zeros", slow_zeros)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Event", Ready)
    results = _parallel_reserve(device)
    assert len(allocations) == len(acknowledgements) == 1
    assert len({t.data_ptr() for t in results}) == (1 if shared_key else 8)
    assert len(host._COUNTERS) == (1 if shared_key else 8)
    assert all(t.untyped_storage().data_ptr() == host._POOL[device].untyped_storage().data_ptr() for t in results)


@pytest.mark.parametrize("ms", [(8, 16), (32, 64), (144, 192)])
def test_two_host_threads_enqueue_independent_streams(ms):
    """Exercise actual thread-local stream keys and concurrent entry calls, after serial warm-up."""
    from . import test_qkv_insert as checks

    checks.require_gfx950()
    _, (weight, scale), _, _ = checks.weights()
    jobs = []
    for m in ms:
        codes, row_major, inputs = checks.make(m, 7200 + m)
        expected = checks.run("chain", m, codes, row_major, inputs)
        checks.run("entry", m, codes, row_major, inputs)
        q, iq = checks.outputs(m)
        jobs.append((m, codes, inputs, expected, (q, iq, inputs[8].clone(), inputs[9].clone())))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    start = threading.Barrier(2)
    pointers = []

    def enqueue(i):
        m, codes, inputs, expected, (q, iq, kv, index) = jobs[i]
        torch.cuda.set_device(0)
        with torch.cuda.stream(streams[i]):
            start.wait(timeout=30)
            pointers.append(host._counters(codes[0].device).data_ptr())
            start.wait(timeout=30)
            assert len(set(pointers)) == 2  # fail before launching kernels with aliased counters
            for _ in range(20):
                kv.copy_(inputs[8])
                index.copy_(inputs[9])
                q.fill_(7)
                iq[:, :128].fill_(0x77)
                host.mxfp8_qkv_norm_rope_nvfp4_insert(
                    *codes, weight, scale, m, *inputs[:8], kv, index, *inputs[10:], q,
                    iq.view(torch.float8_e4m3fn)[:, :128], **checks.KW,
                )
            streams[i].synchronize()
            assert checks.nbytes_differ((q, iq, kv, index), expected) == [0] * 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(enqueue, range(2)))
