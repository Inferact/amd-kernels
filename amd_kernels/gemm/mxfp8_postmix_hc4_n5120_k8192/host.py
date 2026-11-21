# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache, row dispatch and entry of the wo_b GEMM with the mHC post-mix epilogue."""

import functools
from dataclasses import dataclass

import torch
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.gemm.mxfp8_common.gemm import preshuffle_mxfp8_weight
from amd_kernels.warm_up import entry

from .kernel import H, K, N, NS, PSLOT, compile_postmix_gemm

MAX_ROWS = 768
N_MIX = 24


@dataclass(frozen=True)
class Config:
    """16 * ``rt`` rows and 16 * ``ni`` columns per workgroup, ``kw`` K-waves, ``ks`` K splits over workgroups (the last
    to arrive runs the epilogue), ``pf`` epilogue prefetch distance in K-steps before the end, ``sched`` scheduling
    barriers, ``w_cm`` the weight cache modifier."""

    rt: int
    ni: int
    kw: int
    ks: int = 1
    pf: int = 0
    sched: int = 1
    w_cm: int | None = None


# (largest M of the row, config): the fastest of the swept (rt, ni, kw, ks) at each bound, graph replay over cold weights
# (``benchmarks/gemm/.../benchmark_chain.py``; the gate variant is swept inside its chain with the finisher)
_ROWS: tuple[tuple[int, Config], ...] = (
    (16, Config(1, 2, 1, 1, w_cm=2)),
    (32, Config(1, 5, 2, 2, w_cm=2)),
    (64, Config(2, 5, 4, 2, w_cm=2)),
    (128, Config(2, 5, 4, 1, w_cm=1)),
    (160, Config(3, 5, 2, 1, w_cm=0)),
    (192, Config(2, 8, 4, 1, w_cm=0)),
    (256, Config(3, 8, 4, 1, w_cm=1)),
    (320, Config(4, 8, 4, 1, w_cm=0)),
    (384, Config(4, 8, 4, 1, w_cm=0)),
    (512, Config(8, 5, 2, 1, w_cm=2)),
    (640, Config(4, 8, 2, 1, w_cm=0)),
    (768, Config(4, 8, 2, 1, w_cm=0)),
)

_GATE_ROWS: tuple[tuple[int, Config], ...] = (
    (16, Config(1, 2, 1, 1, w_cm=2)),
    (32, Config(1, 5, 2, 2, w_cm=2)),
    (64, Config(2, 5, 2, 2, w_cm=2)),
    (128, Config(2, 5, 4, 1, w_cm=1)),
    (160, Config(2, 8, 4, 1, w_cm=0)),
    (192, Config(2, 8, 2, 1, w_cm=0)),
    (256, Config(3, 8, 4, 1, w_cm=1)),
    (320, Config(4, 8, 4, 1, w_cm=0)),
    (384, Config(4, 8, 4, 1, w_cm=0)),
    (512, Config(3, 8, 2, 1, w_cm=0)),
    (640, Config(4, 8, 2, 1, w_cm=0)),
    (768, Config(4, 8, 2, 1, w_cm=1)),
)


def select_gates(m: int) -> Config:
    """The config of a row count for the epilogue that also writes the next seam's gate partials."""
    assert 1 <= m <= MAX_ROWS, m
    return next(c for bound, c in _GATE_ROWS if m <= bound)


def select(m: int) -> Config:
    """The config of a row count."""
    assert 1 <= m <= MAX_ROWS, m
    return next(c for bound, c in _ROWS if m <= bound)


@functools.cache
def _compiled(cfg: Config, gates: bool = False, aux: bool = False):
    return compile_postmix_gemm(
        RT=cfg.rt, NI=cfg.ni, KW=cfg.kw, KS=cfg.ks, PF=cfg.pf, GATES=gates, AUX=aux, sched_barriers=bool(cfg.sched),
        W_CM=cfg.w_cm,
    )  # fmt: skip


COUNTERS = 1 << 16
_WORKSPACES: dict[torch.device, list] = {}


def _partial_bytes(cfg: Config, m: int) -> int:
    return -(-m // (16 * cfg.rt)) * (N // (16 * cfg.ni)) * cfg.ks * cfg.rt * cfg.ni * 1024 if cfg.ks > 1 else 0


def _workspace(device: torch.device, nbytes: int):
    """Split-K partials and arrival counters, one set per device (launches on one stream); the counters are zeroed
    once and every launch leaves them at zero. Allocated before graph capture (``warmup_postmix_gemm``)."""
    ws = _WORKSPACES.get(device)
    if ws is None or ws[0].numel() * 4 < nbytes:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("call warmup_postmix_gemm before CUDA-graph capture")
        counters = ws[1] if ws else torch.zeros(COUNTERS, dtype=torch.int32, device=device)
        ws = _WORKSPACES[device] = [torch.empty(max(nbytes, 16) // 4, dtype=torch.float32, device=device), counters]
    return ws


def preshuffle_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes [5120, 8192] and E8M0 scales [5120, 256] -> the layouts the GEMM reads (load time)."""
    assert tuple(weight.shape) == (N, K) and weight.dtype == torch.float8_e4m3fn
    assert tuple(weight_scale.shape) == (N, K // 32) and weight_scale.dtype == torch.uint8
    return preshuffle_mxfp8_weight(weight, weight_scale)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


def pack_gate_weight(fn: torch.Tensor) -> torch.Tensor:
    """fp32 ``fn`` [24, 4 * 5120] -> the epilogue's bf16 hi / lo B operands (once, at load time): per (16-column
    group, stream) four 512 B operands (hi and lo of mixes 0-15, hi and lo of mixes 16-23), lane (q, n) holding the 4
    columns 4q .. 4q + 3 of mix n."""
    assert fn.shape == (N_MIX, NS * H) and fn.dtype == torch.float32 and fn.is_contiguous()
    with torch.no_grad():
        f = fn.view(N_MIX, NS, H // 16, 4, 4)
        hi = f.to(torch.bfloat16)
        lo = (f - hi.float()).to(torch.bfloat16)

        def lanes(t, first):
            x = t[first : first + 16]
            return torch.cat([x, x.new_zeros((16 - x.shape[0],) + x.shape[1:])]) if x.shape[0] < 16 else x

        ops = torch.stack([lanes(hi, 0), lanes(lo, 0), lanes(hi, 16), lanes(lo, 16)])  # [op, n, s, cg, q, i]
        return ops.permute(3, 2, 0, 4, 1, 5).contiguous()  # [cg, s, op, q, n, i]


def _check(codes, scales, residual, post_mix, comb_mix, out):
    m = codes.shape[0]
    assert 1 <= m <= MAX_ROWS, f"rows {m} outside 1..{MAX_ROWS}"
    assert codes.dtype == torch.float8_e4m3fn and codes.is_contiguous() and tuple(codes.shape) == (m, K)
    assert scales.dtype in (torch.uint8, torch.float8_e8m0fnu) and scales.is_contiguous()
    assert scales.numel() == m * (K // 32)
    assert residual.dtype == torch.bfloat16 and residual.is_contiguous() and tuple(residual.shape) == (m, NS, H)
    assert post_mix.dtype == comb_mix.dtype == torch.float32 and post_mix.is_contiguous() and comb_mix.is_contiguous()
    assert post_mix.numel() == m * NS and comb_mix.numel() == m * NS * NS
    if out is None:
        out = torch.empty_like(residual)
    assert out.shape == residual.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out.untyped_storage().data_ptr() != residual.untyped_storage().data_ptr()
    return out


def _launch(cfg, codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out, gates=None):
    m = codes.shape[0]
    exe = _compiled(cfg, gates is not None, gates is not None and gates[3] is not None)
    tiles = -(-m // (16 * cfg.rt)) * exe.n_blocks
    assert tiles <= COUNTERS
    part, counters = _workspace(codes.device, _partial_bytes(cfg, m))
    dummy = torch.empty(4, dtype=torch.int32, device=codes.device)
    fn, pre, x1, mean, gp = gates if gates is not None else (dummy, dummy, dummy, None, dummy)
    _run_compiled(
        exe, _u8(codes), _u8(weight), _u8(residual), _u8(out), _u8(scales), _u8(weight_scale), _u8(post_mix),
        _u8(comb_mix), _u8(part), _u8(counters), _u8(fn), _u8(pre), _u8(x1), _u8(dummy if mean is None else mean),
        _u8(gp), m, scales.numel(), tiles * cfg.ks, torch.cuda.current_stream(),
    )  # fmt: skip


def _postmix_gemm(codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out=None, *, config=None):
    out = _check(codes, scales, residual, post_mix, comb_mix, out)
    _launch(config or select(codes.shape[0]), codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out)
    return out


def _postmix_gates_gemm(
    codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix, fn_packed, *, aux_mean=False,
    config=None,
):  # fmt: skip
    m = codes.shape[0]
    out = _check(codes, scales, residual, post_mix, comb_mix, None)
    assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous() and pre_mix.numel() == m * NS
    assert fn_packed.dtype == torch.bfloat16 and fn_packed.shape == (H // 16, NS, 4, 4, 16, 4)
    cfg = config or select_gates(m)
    nb = N // (16 * cfg.ni) * cfg.kw
    x1 = torch.empty(m, H, dtype=torch.bfloat16, device=codes.device)
    mean = torch.empty(m, H, dtype=torch.bfloat16, device=codes.device) if aux_mean else None
    gp = torch.empty(m, nb, PSLOT, dtype=torch.float32, device=codes.device)
    _launch(cfg, codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out,
            (fn_packed, pre_mix, x1, mean, gp))  # fmt: skip
    return out, x1, gp, mean


def _mode(*args, **kwargs):
    return ()


def _sizes(mode):
    first, prev = [], 0
    for bound, _ in _ROWS:
        first.append(prev + 1)
        prev = bound
    return tuple(first)


def _inputs(mode, n, codes, scales, weight, weight_scale, residual, post_mix, comb_mix, out=None):
    dev = codes.device
    return (
        torch.zeros(n, K, dtype=torch.float8_e4m3fn, device=dev), torch.zeros(n, K // 32, dtype=torch.uint8, device=dev),
        weight, weight_scale, torch.zeros(n, NS, H, dtype=torch.bfloat16, device=dev),
        torch.zeros(n, NS, device=dev), torch.zeros(n, NS, NS, device=dev),
    ), {}  # fmt: skip


postmix_gemm = entry("gemm/mxfp8_postmix_hc4_n5120_k8192", mode=_mode, sizes=_sizes, inputs=_inputs)(_postmix_gemm)
postmix_gemm.__doc__ = """``R'[m, j] = bf16(p[m, j] * y[m] + sum_h c[m, h, j] * R[m, h])`` with ``y = bf16(x @ W^T)`` for
pre-quantized
``x`` (E4M3 codes [M, 8192] + row-major E8M0 scales [M, 256]), the preshuffled wo_b weight, the residual ``R``
bf16 [M, 4, 5120] and the gates ``post`` [M, 4, 1] / ``comb`` [M, 4, 4] fp32 of the seam that produced ``R``
(``reference.py``). Returns the new residual bf16 [M, 4, 5120], the next seam's input as it is. CUDA-graph safe,
deterministic; 1 <= M <= ``MAX_ROWS``."""


def _gates_mode(*args, aux_mean=False, **kwargs):
    return (bool(aux_mean),)


def _gates_sizes(mode):
    return tuple(r[0] for r in _GATE_ROWS)


def _gates_inputs(mode, n, codes, scales, weight, weight_scale, residual, post_mix, comb_mix, pre_mix, fn_packed, **kw):
    args, _ = _inputs(mode, n, codes, scales, weight, weight_scale, residual, post_mix, comb_mix)
    return (*args, torch.zeros(n, NS, device=codes.device), fn_packed), dict(aux_mean=mode[0])


postmix_gates_gemm = entry(
    "gemm/mxfp8_postmix_hc4_n5120_k8192/gates", mode=_gates_mode, sizes=_gates_sizes, inputs=_gates_inputs
)(_postmix_gates_gemm)

postmix_gates_gemm.__doc__ = """``postmix_gemm`` that also writes what the next seam's finisher
(``norm/mhc_gate_finish_rmsnorm_h5120``) needs, so the gate projection is never redone: with the carried pre gate
``pre_mix`` [M, 4] fp32 and ``pack_gate_weight(fn)``, returns (new residual bf16 [M, 4, 5120], collapse ``x1`` bf16
[M, 5120], records fp32 [M, NB * KW, 32] (``reference.py``: the sums over the records are the gate projection, the sum
of squares of the residual and of x1), mean of the four new streams bf16 [M, 5120] or None with ``aux_mean``)."""


def warmup_postmix_gemm(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    """Compile every row's variants (plain, gate partials, gate partials with the aux mean) and allocate the split-K
    workspace before graph capture."""
    dev = weight.device
    fn = pack_gate_weight(torch.zeros(N_MIX, NS * H, dtype=torch.float32, device=dev))
    codes = torch.zeros(1, K, dtype=torch.float8_e4m3fn, device=dev)
    for m in _sizes(()):
        args, _ = _inputs((), m, codes, None, weight, weight_scale, None, None, None)
        postmix_gemm(*args)
    for aux in (False, True):
        for m in _gates_sizes(()):
            args, _ = _inputs((), m, codes, None, weight, weight_scale, None, None, None)
            postmix_gates_gemm(*args, torch.zeros(m, NS, device=dev), fn, aux_mean=aux)
    torch.cuda.synchronize(dev)
