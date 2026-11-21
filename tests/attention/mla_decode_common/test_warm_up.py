# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The single-GPU entries in a fresh process with an empty FlyDSL cache: the first call of each mode through the
entry (or ``<op>_kernel`` + a compile-only warm-up, as a caller prewarms) warms up the routed package; after
``compile_watch.arm`` no served row count compiles; a CUDA graph of each entry call, captured with
``importlib.import_module`` failing (the entry imports nothing at that point), replays the eager result byte for
byte."""

import importlib
import json
import os
import subprocess
import sys

import torch

from .cases import require_gfx950

SERVED_M = (1, 6, 12, 24, 40, 65, 80, 128, 192)
SPARSE_ROWS = (5, 10, 12, 20, 24, 40, 42, 43, 65, 80, 128, 192)
DENSE_REQS = (1, 2, 5, 13, 33, 48, 63)


def prewarm(fn, *args, **kwargs):
    from amd_kernels import warm_up

    spec = fn.entry
    key = (spec.name, spec.mode(*args, **kwargs))
    if key not in warm_up.warmed():
        warm_up._warm(spec, key, args, kwargs)


def captured(call, outputs):
    """Replay of a graph of ``call()`` (captured with imports failing) against its eager result: True when equal."""
    from .cases import same_bytes, tensors

    eager = [t.clone() for t in tensors(call())]
    for t in tensors(outputs):
        t.zero_()
    graph, real = torch.cuda.CUDAGraph(), importlib.import_module

    def no_import(name, package=None):
        raise AssertionError(f"import of {name} inside capture")

    importlib.import_module = no_import
    try:
        with torch.cuda.graph(graph):
            got = call()
    finally:
        importlib.import_module = real
    graph.replay()
    torch.cuda.synchronize()
    return all(same_bytes(a, b) for a, b in zip(tensors(got), eager))


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.attention import mla_decode_common as E
    from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import quantize_q

    from tests.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.cases import make_batch as sparse_batch
    from tests.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.cases import served_counts

    from .test_dense import case as dense_case
    from .test_prologue import insert_args, prep_args

    warm_up.set_auto(True)
    gen = torch.Generator().manual_seed(0)
    prewarm(E.qkv_a_norm_kv_insert_kernel(*insert_args(1, "index", gen)), *insert_args(1, "index", gen))
    for form in ("norm", "insert"):
        E.qkv_a_norm_kv_insert(*insert_args(1, form, gen))
    prewarm(E.indexer_prep_kernel(*prep_args(1, False, gen)), *prep_args(1, False, gen))
    E.indexer_prep(*prep_args(1, True, gen))
    q, cache, indptr, slots = sparse_batch(served_counts(192, gen), gen)
    E.fp8_sparse_mla_decode(q[:1], cache, indptr[:2], slots[: int(indptr[1])], 0.1)
    prewarm(E.fp8_sparse_mla_decode_kernel(None, cache, indptr[:2], slots, 0.1, q8=quantize_q(q[:1])),
            None, cache, indptr[:2], slots, 0.1, q8=quantize_q(q[:1]))
    for qlen in (1, 3, 4):
        args, _ = dense_case(qlen, 1, False)
        E.fp8_dense_mla_decode(*args)
    torch.cuda.synchronize()
    warmed = len(warm_up.warmed())

    late = []
    compile_watch.arm(late.append)
    for n in SERVED_M:
        for form in ("norm", "insert", "index"):
            E.qkv_a_norm_kv_insert(*insert_args(n, form, gen))
        for keys in (False, True):
            E.indexer_prep(*prep_args(n, keys, gen))
    q8 = quantize_q(q)
    for rows in SPARSE_ROWS:
        s = slots[: int(indptr[rows])]
        E.fp8_sparse_mla_decode(q[:rows], cache, indptr[: rows + 1], s, 0.1)
        E.fp8_sparse_mla_decode(None, cache, indptr[: rows + 1], s, 0.1, q8=q8[:rows])
    for qlen in (1, 3, 4):
        for reqs in DENSE_REQS:
            args, _ = dense_case(qlen, reqs, False)
            E.fp8_dense_mla_decode(*args)
    torch.cuda.synchronize()

    graphs = {}
    for form in ("norm", "insert", "index"):
        a = insert_args(65, form, gen)
        outs = [a[4], a[5]] + ([a[8]["weights"]] if form == "index" else [])
        graphs[f"insert_{form}"] = captured(lambda a=a, outs=outs: (E.qkv_a_norm_kv_insert(*a), outs)[1], outs)
    for keys in (False, True):
        a = prep_args(65, keys, gen)
        outs = [a[8], a[10], a[11]] + ([a[12]] if keys else [])
        graphs[f"prep_{keys}"] = captured(lambda a=a, outs=outs: (E.indexer_prep(*a), outs)[1], outs)
    for rows in (12, 128):
        out = torch.zeros((rows, 96, 512), dtype=torch.bfloat16, device="cuda")
        args = (q[:rows], cache, indptr[: rows + 1], slots[: int(indptr[rows])], 0.1)
        graphs[f"sparse_{rows}"] = captured(lambda args=args, out=out: E.fp8_sparse_mla_decode(*args, out=out), [out])
    for qlen, reqs in ((1, 63), (4, 13)):
        args, kw = dense_case(qlen, reqs, True)
        graphs[f"dense_{qlen}_{reqs}"] = captured(lambda args=args, kw=kw: E.fp8_dense_mla_decode(*args, **kw),
                                                  list(kw.values()))
    torch.cuda.synchronize()
    return {"warmed": warmed, "late": late, "count": compile_watch.late_compiles(), "graphs": graphs,
            "after": len(warm_up.warmed())}


def test_warm_up_then_no_compile_and_graph_replay(tmp_path):
    require_gfx950()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=900, cwd=root,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    # 3 insert modes + 2 prep modes + 2 sparse modes + 3 dense modes
    assert r["warmed"] == 10 and r["after"] == 10, r
    assert r["late"] == [] and r["count"] == 0, r
    assert all(r["graphs"].values()) and len(r["graphs"]) == 9, r["graphs"]


if __name__ == "__main__":
    print(json.dumps(scenario()))
