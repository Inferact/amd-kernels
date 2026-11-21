# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 through the dense MLP entry (a fresh process and FlyDSL cache): the first call warms the routed package up,
every served row count then compiles nothing, and a CUDA graph of the entry call imports and allocates nothing and
replays the eager bits."""
import importlib
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.decoder.bf16_dense_mlp_common import (bf16_dense_mlp, bf16_dense_mlp_shapes,
                                                           dense_mlp_pack_down, dense_mlp_pack_gate_up,
                                                           dense_mlp_rows_buffer, dense_mlp_workspace, host as entry)

    warm_up.set_auto(True)
    dev = torch.device("cuda")
    result = {}
    for key in sorted(bf16_dense_mlp_shapes()):
        hidden, inter = key
        host = entry._shape_host(key)
        ws = dense_mlp_workspace(dev, hidden=hidden, inter=inter)
        gu = dense_mlp_pack_gate_up((torch.randn(2 * inter, hidden, device=dev) * 0.02).bfloat16())
        dn = dense_mlp_pack_down((torch.randn(hidden, inter, device=dev) * 0.03).bfloat16())
        bf16_dense_mlp(ws, dense_mlp_rows_buffer(16, dev, down=dn).zero_(), gu, dn, 16)
        torch.cuda.synchronize()
        warns = []
        compile_watch.arm(warns.append)
        for rows in range(1, host.MAX_ROWS + 1):
            xp = dense_mlp_rows_buffer(rows, dev, down=dn).normal_()
            bf16_dense_mlp(ws, xp, gu, dn, rows)
        torch.cuda.synchronize()
        graph = []
        for rows in (1, host.MAX_ROWS):
            xp = dense_mlp_rows_buffer(rows, dev, down=dn).normal_()
            out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
            eager = bf16_dense_mlp(ws, xp, gu, dn, rows, out).clone()
            torch.cuda.synchronize()
            out.zero_()
            real = importlib.import_module
            importlib.import_module = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("import in capture"))
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
                    bf16_dense_mlp(ws, xp, gu, dn, rows, out)
                    new_allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
            finally:
                importlib.import_module = real
            g.replay()
            torch.cuda.synchronize()
            graph.append({"rows": rows, "allocs": new_allocs, "equal": torch.equal(out, eager)})
        result[str(key)] = {"late": compile_watch.late_compiles(), "warns": warns[:2],
                            "entry_late": warm_up.late_compiles(), "graph": graph}
    return result


def test_warm_up_then_no_compile_and_graph(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    for key, r in result.items():
        assert r["late"] == 0 and r["entry_late"] == 0, (key, r)
        assert all(g["allocs"] == 0 and g["equal"] for g in r["graph"]), (key, r)


if __name__ == "__main__":
    print(json.dumps(scenario()))
