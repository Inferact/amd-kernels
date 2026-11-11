# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

The op launches one FlyDSL kernel, compiled on its first launch per ``host._get_kernel(splits, topk, xcds, quant,
exact_fp16, tail)`` key (``_run_compiled``); the key follows from the shapes alone (``plan.py``). Per served
configuration and output mode, ``test_no_compile_after_warmup`` warms one request count per key, then serves every
other request count of the range: none may compile, and each launch's key must be the plan's.

CUDA graph capture alone compiles what pure-decode steps launch (the MXFP8 output, ``quant_out``). A mixed prefill +
decode step runs its decode rows eagerly with the bf16 output, other keys: ``test_mixed_step_after_decode_capture`` is
a strict xfail, which ``host.warm_up_decode`` closes (``test_warm_up_entry.py``).
"""

import time

import flydsl.compiler as flyc
import pytest
import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import GROUP, HEAD_DIM

from .plan import SERVED, XCDS, plan, variants
from .test_sparse_decode import lengths_for, make_case, quant_buffers, require_gfx950, run

MIXED_PROBES = 4


class Launches:
    """Counts ``flyc.compile`` calls and records the host's kernel keys."""

    def __init__(self, monkeypatch):
        self.compiled, self.keys = [], []
        real_compile, real_get = flyc.compile, host._get_kernel

        def counting(fn, *args, **kwargs):
            self.compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
            return real_compile(fn, *args, **kwargs)

        def spy(*args):
            self.keys.append(args)
            return real_get(*args)

        monkeypatch.setattr(flyc, "compile", counting)
        monkeypatch.setattr(host, "_get_kernel", spy)


class Server:
    """One case at the configuration's largest batch; a launch for ``reqs`` requests takes its first ``reqs``
    requests (graph padding: the last request, length 0)."""

    def __init__(self, config):
        hkv, self.compute, self.ql, self.max_reqs = SERVED[config]
        self.hkv = hkv
        self.case = make_case(hkv, self.ql, lengths_for(self.max_reqs, 5, pad=1), seed=5)
        rows = self.max_reqs * self.ql
        self.width = hkv * GROUP * HEAD_DIM
        self.out = torch.empty_like(self.case["q"])
        self.codes, self.scales = quant_buffers(rows, self.width)

    def launch(self, reqs, quant):
        rows = reqs * self.ql
        c = dict(self.case)
        c["q"] = c["q"][:rows]
        c["topk_idx"] = c["topk_idx"][:, :rows]
        c["block_table"] = c["block_table"][:reqs]
        c["seq_lens"] = c["seq_lens"][:reqs]
        q = (self.codes, self.scales, 16) if quant else None
        run(c, self.compute, out=self.out[:rows], quant=q, xcd_count=XCDS)

    def key(self, reqs, quant):
        return plan(reqs, self.ql, self.hkv, self.compute, quant)


CASES = [(c, q) for c in SERVED for q in (False, True)]


@pytest.mark.parametrize(
    "config,quant", CASES, ids=[f"{c}-{'mxfp8' if q else 'bf16'}" for c, q in CASES]
)
@torch.inference_mode()
def test_no_compile_after_warmup(config, quant, monkeypatch):
    require_gfx950()
    host._get_kernel.cache_clear()  # compile again what earlier tests compiled
    server = Server(config)
    warm = variants(config, quant)
    launches = Launches(monkeypatch)
    seconds = []
    for key, reqs in warm.items():
        t0 = time.perf_counter()
        server.launch(reqs, quant)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
        assert launches.keys[-1] == key, (reqs, launches.keys[-1], key)
    assert len(launches.compiled) == len(warm), launches.compiled
    print(
        f"{config} {'mxfp8' if quant else 'bf16'}: {len(warm)} variants over "
        f"1..{server.max_reqs} requests {sorted(warm.items(), key=lambda kv: kv[1])}; "
        f"first launch {min(seconds):.2f}-{max(seconds):.2f} s"
    )
    launches.compiled.clear()
    for reqs in range(1, server.max_reqs + 1):
        if reqs in warm.values():
            continue
        server.launch(reqs, quant)
        assert launches.keys[-1] == server.key(reqs, quant), reqs
        assert not launches.compiled, (
            f"R={reqs} (key {launches.keys[-1]}) compiled {launches.compiled} after "
            f"warm-up on {sorted(warm.values())}"
        )


@pytest.mark.xfail(
    strict=True,
    reason="decode CUDA graph capture alone compiles the MXFP8-output variants of pure-decode steps; the "
    "bf16-output variants a mixed prefill + decode step launches (quant_out None) compile on the forward path",
)
@pytest.mark.parametrize("config", list(SERVED))
@torch.inference_mode()
def test_mixed_step_after_decode_capture(config, monkeypatch):
    require_gfx950()
    host._get_kernel.cache_clear()
    server = Server(config)
    for reqs in range(1, server.max_reqs + 1):  # every pure-decode graph size
        server.launch(reqs, quant=True)
    torch.cuda.synchronize()
    launches = Launches(monkeypatch)
    step = max(1, server.max_reqs // MIXED_PROBES)
    for reqs in range(1, server.max_reqs + 1, step):  # a mixed step's decode rows
        server.launch(reqs, quant=False)
        assert not launches.compiled, (
            f"{config}: a mixed step with {reqs} decode requests compiled "
            f"{launches.compiled} (key {launches.keys[-1]}) after every pure-decode size"
        )


@pytest.mark.parametrize('quant',[False,True])
def test_fp8_warmup_includes_union(monkeypatch,quant):
    """Every fp8 split tier and the union compile before the size probes. The per-row kernel overlaps its first
    block with the table only at <= one workgroup per CU (``_prefetch``): 160 (four splits), 320 / 384 (four / two tail
    splits) and 512 rows (one split) warm the others."""
    from .test_fp8_decode import FP8, unit_scales

    require_gfx950()
    def launch(rows):
        case=unit_scales(make_case(1,8,lengths_for(rows//8,seed=31),seed=31))
        kw={'quant':(*quant_buffers(rows,2048),16)} if quant else {}
        run(case,FP8,**kw)
    for rows in [8,64,128,160,256,320,384,512,1024]:launch(rows)
    compiled=[]
    real=flyc.compile
    def count(*a,**kw):
        compiled.append(1);return real(*a,**kw)
    monkeypatch.setattr(flyc,'compile',count)
    for rows in [16,24,32,40,136,192,200,264,328,376,392,520,576,584,640,648,768,776,1032,2048]:launch(rows)
    assert not compiled
