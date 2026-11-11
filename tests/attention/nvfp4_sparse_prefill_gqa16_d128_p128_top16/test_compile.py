# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

A served configuration (compute dtype, KV heads per GPU, output mode) launches one kernel per call, picked on the host
by the token count T: the combine's rounds (two once ``ceil(T / 4) Hkv`` exceeds the CU count) and the small-T tiles.
``variant`` is the key of the host's compile cache (``host._get_kernel``). Per configuration: warm one T per variant
over the served range, then probe other Ts; a probe that compiles means T reaches a compile key on the forward path.
"""

import dataclasses
import time

import flydsl.compiler as flyc
import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import pack_query
from .cases import TP4_HKV, make_case, require_gfx950
from .test_sparse_prefill import case_slice, run, run_pack, run_quant

# served prefill tokens per step: max-num-batched-tokens 65536 (TP4)
RANGES = {TP4_HKV: range(1, 65537)}
PROBES = 6
# (compute dtype, KV heads, output mode): pack is the sequence-parallel TP4 prefill's
CONFIGS = [(variant, TP4_HKV, mode) for variant in ("bf16", "fp16") for mode in ("plain", "quant", "pack")]


def variant_key(variant, hkv, mode, t):
    """``host._get_kernel``'s key for a call of ``t`` tokens (bf16 Q, ``sched=False``)."""
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    rounds = 2 if (t + 3) // 4 * hkv > cus else 1
    if variant in ("bf16", "fp16") and hkv == TP4_HKV and t <= 1536:
        return (variant, mode == "pack", rounds, mode == "quant", False, False, 1 if t <= 256 else 2 if t <= 1025 else 3, 512 < t <= 1025)
    return (variant, mode == "pack", rounds, mode == "quant", False, False)


def serve(case, variant, mode, t):
    """One served step of ``t`` tokens, as the caller makes it."""
    sub = case_slice(case, 0, t)
    if mode == "plain":
        run(sub, variant)
    elif mode == "quant":
        run_quant(sub, variant)
    else:
        run_pack(sub, variant)


def warm_and_probe(served, key):
    warm = {}
    for t in served:
        warm.setdefault(key(t), t)
    rest = [t for t in served if t not in warm.values()]
    return warm, rest[:: max(1, len(rest) // PROBES)][:PROBES]


def count_compiles(monkeypatch):
    compiled = []
    real_compile = flyc.compile

    def counting_compile(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", getattr(fn, "__name__", type(fn).__name__)))
        return real_compile(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting_compile)
    return compiled


@pytest.fixture(scope="module")
def cases():
    require_gfx950()
    return {
        hkv: make_case([r[-1]], [r[-1] + 2048], hkv, seed=0, picks="shared")
        for hkv, r in RANGES.items()
    }


def _boundary_ts(served, key):
    """The token counts where the key changes, then the rest."""
    edges = [t for t in served[1:] if key(t) != key(t - 1)]
    return sorted({served[0], *edges, *(e - 1 for e in edges)})


@pytest.mark.parametrize("variant,hkv,mode", CONFIGS, ids=["-".join(map(str, c)) for c in CONFIGS])
@torch.inference_mode()
def test_no_compile_after_warmup(variant, hkv, mode, cases, monkeypatch):
    require_gfx950()
    served = RANGES[hkv]
    key = lambda t: variant_key(variant, hkv, mode, t)  # noqa: E731
    warm, probe = warm_and_probe(served, key)
    probe = sorted({*probe, *(t for t in _boundary_ts(served, key) if t not in warm.values())})
    case = cases[hkv]
    for t in warm.values():
        start = time.perf_counter()
        serve(case, variant, mode, t)
        torch.cuda.synchronize()
        print(f"warm {key(t)} at T={t}: {time.perf_counter() - start:.2f} s")
    compiled = count_compiles(monkeypatch)
    for t in probe:
        serve(case, variant, mode, t)
        assert not compiled, (
            f"T={t} (variant {key(t)}) compiled {compiled} after warm-up on "
            f"{sorted(warm.items(), key=str)}"
        )


@pytest.mark.parametrize("mode", ["plain", "pack", "quant"])
@pytest.mark.parametrize("prepacked", [False, True])
def test_fp8_no_compile_after_warmup(mode, prepacked, monkeypatch):
    require_gfx950()
    case = make_case([32768], [32768 + 2048], TP4_HKV, seed=71, picks="shared")
    if prepacked:
        case = dataclasses.replace(case, q=pack_query(case.q))
    for t in [1, 512, 513, 1025, 1088]:
        serve(case, "fp8", mode, t)
    compiled = count_compiles(monkeypatch)
    for t in [3, 256, 257, 511, 514, 1024, 1087, 1536, 1537, 2048, 4096, 8192, 32768]:
        serve(case, "fp8", mode, t)
        assert not compiled
