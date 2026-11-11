# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``init`` compiles every variant a TP4 server (16 q heads, 1 KV / index head, 8 rows per request, 1-256 requests,
1M context) launches through the layer: afterwards no step compiles: the index top-16 + decode block and the decode
alone at 1-64 and up to 256 requests (32K / 1M context, bf16 and MXFP8 output from ``QUANT_OUT_MIN_ROWS`` rows), the
prefill and the prefill indexer at 1-32768 tokens (indexer: up to 1M context) and the insert's prefill tiles at
1537-40000 tokens."""
import json
import os
import subprocess
import sys

import torch

from tests.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.test_sparse_decode import require_gfx950

QL, HMAX, MAX_REQS = 8, 1 << 20, 256


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels import warm_up
    from amd_kernels.attention import nvfp4_sparse_layer_gqa16_d128_p128_top16 as layer
    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import norm_rope_nvfp4_insert
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import prefill_top16

    torch.set_num_threads(8)
    warm_up.set_auto(True)
    count = []
    original = MlirCompiler.compile

    def spy(*a, **kw):
        count.append(1)
        return original(*a, **kw)

    MlirCompiler.compile = spy
    dev = torch.device("cuda")
    index = layer.init(dev, num_heads=16, num_kv_heads=1, num_index_heads=1, query_len=QL, max_reqs=MAX_REQS,
                       max_seq_len=HMAX)
    warm, late = len(count), []

    def probe(tag, fn):
        before = len(count)
        fn()
        torch.cuda.synchronize()
        if len(count) != before:
            late.append(tag)

    z = dict(device=dev)
    one = torch.ones((), dtype=torch.float32, **z)
    icache = torch.zeros(2, 128, 72, dtype=torch.uint8, **z)
    kv = torch.zeros(2, 2, 128, 72, dtype=torch.uint8, **z)
    for n in list(range(1, 65)) + [65, 100, 128, 129, 200, 256]:
        rows = n * QL
        for ctx in ((32768, HMAX) if n in (1, 8, 32, 33, 64, 256) else (HMAX,)):
            for quant in (False, True):
                qo = layer.quant_out_buffers(rows, 6144, 2048, dev) if quant else None
                if quant and qo is None:
                    continue
                iq = torch.zeros(rows, 1, 128, dtype=torch.float8_e4m3fn, **z)
                q = torch.zeros(rows, 16, 128, dtype=torch.bfloat16, **z)
                table = torch.zeros(n, ctx // 128, dtype=torch.int32, **z)
                seq = torch.full((n,), ctx, dtype=torch.int32, **z)
                ids = torch.full((1, rows, 16), -1, dtype=torch.int32, **z)
                ids[..., 0] = 0
                probe(("decode", n, ctx, quant), lambda: layer.sparse_decode(
                    q, kv, ids, table, seq, 128**-0.5, torch.empty_like(q), k_scale=one, v_scale=one,
                    decode_query_len=QL, quant_out=qo))
                probe(("block", n, ctx, quant), lambda: layer.index_sparse_decode(
                    index, iq, icache, table, seq, 0, 1, q, kv, ids, table, seq, 128**-0.5, torch.empty_like(q),
                    k_scale=one, v_scale=one, decode_query_len=QL, max_seq_len=ctx, quant_out=qo))
    for t in (1, 8, 100, 512, 513, 1500, 2048, 2049, 4096, 16384, 32768):
        q = torch.zeros(t, 16, 128, dtype=torch.bfloat16, **z)
        topk = torch.full((1, t, 16), -1, dtype=torch.int32, **z)
        topk[..., 0] = 0
        zeros = torch.zeros(t, dtype=torch.int32, **z)
        probe(("prefill", t), lambda: layer.sparse_prefill(
            q, kv, topk, torch.zeros(1, 1, dtype=torch.int32, **z), zeros, zeros, 1, 128**-0.5, torch.empty_like(q),
            one, one))
        for ctx in sorted({t, 65536 + t, 212992 + t, HMAX} if t <= 16384 else {t, 262144}):
            iq = torch.zeros(t, 1, 128, dtype=torch.float8_e4m3fn, **z)
            table = torch.zeros(1, -(-ctx // 128), dtype=torch.int32, **z)
            cu = torch.tensor([0, t], dtype=torch.int32, **z)
            seq = torch.tensor([ctx], dtype=torch.int32, **z)
            nvp = torch.ones(t, dtype=torch.int32, **z)
            probe(("index prefill", t, ctx), lambda: prefill_top16.index_prefill_top16(
                iq, icache, table, cu, seq, 0, 1, torch.empty(1, t, 16, dtype=torch.int32, **z), max_query_len=t,
                max_seq_len=ctx, num_valid_pages=nvp))
    w = torch.zeros(128, dtype=torch.bfloat16, **z)
    for t in (1537, 1600, 4096, 9000, 24576, 32768, 40000):
        slots = torch.full((t,), -1, dtype=torch.int64, **z)
        probe(("insert", t), lambda: norm_rope_nvfp4_insert(
            torch.zeros(t, 20 * 128, dtype=torch.bfloat16, **z), w, w, w, w,
            torch.zeros(t, 64, dtype=torch.bfloat16, **z), torch.arange(t, **z), slots, slots,
            torch.zeros(1, 2, 128, 72, dtype=torch.uint8, **z), torch.zeros(1, 128, 72, dtype=torch.uint8, **z), one,
            one, torch.empty(t, 16 * 128, dtype=torch.bfloat16, **z),
            torch.empty(t, 128, dtype=torch.float8_e4m3fn, **z), num_heads=16, num_kv_heads=1, num_index_heads=1,
            eps=1e-6))
    return dict(warm=warm, late=late, reported=warm_up.late_compiles())


def test_init_then_no_compile(tmp_path):
    require_gfx950()
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=3000,
                          env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                               "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warm"] > 0 and not result["late"] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
