# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV MXFP8 GEMM + norm / RoPE / NVFP4 inserts, bitwise the deferred chain: ``mxfp8_gemm_shuffled(
defer_reduce=True)``, then ``splitk_reduce_norm_rope_nvfp4_insert`` (split rows) or ``norm_rope_nvfp4_insert``.

One launch (``kernel``) at 1-16 rows; the deferred chain's two launches at 17-768, its GEMM storing non-temporally
on the 160 / 192 / 224 x 48 rows (257-448). Counter storage is separated by stream and graph capture.
"""

import ctypes
import functools
import struct
import threading

import torch

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.gemm.mxfp8_common import gemm as _gemm
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import W_CACHE_MOD, compile_mxfp8_gemm_decode
from amd_kernels.warm_up import entry

from . import kernel as _kernel

N, K = _kernel.N, _kernel.K
MAX_ROWS = 768  # the shuffled-code producer's range (``prequant_code_tile_rows``)
# one launch to here: -4..-11% at 1-16 rows against the deferred chain, slower above (benchmark_chain.py)
FUSED_MAX_ROWS = 16


@functools.cache
def _plan(m: int):
    """(RT, NI, KW, KS, LA) of the deferred GEMM row at ``m`` rows: the one-launch kernel's tile and K split."""
    table = _gemm._select_prequant_config(m, N, K, torch.bfloat16)
    config = _gemm._deferred_qkv_config(m, N, K, table)
    assert config.kernel == _gemm._DECODE_KERNEL and config.split_k > 1 and m <= config.tile_m, (m, config)
    la = (config.la_w, config.la_a) if config.la_w else None
    return config.tile_m // 16, config.tile_n // 16, config.waves_m, config.split_k, la


@functools.cache
def _fused_launcher(plan):
    return _kernel.compile_qkv_fused(*plan)


# decode tiles whose bf16 QKV goes non-temporal (same bytes): chain -0.1..-0.6 us at 257-448, slower elsewhere
# (benchmark_chain.py)
NT_OUT_TILES = ((160, 48), (192, 48), (224, 48))


@functools.cache
def _nt_gemm_launcher(config):
    host = _gemm._shape_host(N, K)
    return compile_mxfp8_gemm_decode(
        N=N, K=K, RT=config.tile_m // 16, NI=config.tile_n // 16, KW=config.waves_m, KS=1,
        LA=(config.la_w, config.la_a) if config.la_w else None, sched_barriers=bool(config.waves_n),
        OUT_CM=W_CACHE_MOD, GROUPED=config in host.GROUPED_ROWS,
    )


@functools.cache
def _nt_config(m):
    """The decode row at ``m`` rows if its output goes non-temporal, else None."""
    config = _gemm._select_prequant_config(m, N, K, torch.bfloat16)
    if config.kernel != _gemm._DECODE_KERNEL or config.split_k != 1:
        return None
    return config if (config.tile_m, config.tile_n) in NT_OUT_TILES else None


_COUNTER_SETS = 512  # eager streams and retained graph captures per device; each graph shares its set serially
_COUNTER_WORDS = _kernel.COUNTERS
_COUNTER_LOCK = threading.Lock()
_POOL: dict[torch.device, torch.Tensor] = {}
_COUNTERS: dict[tuple[torch.device, tuple], torch.Tensor] = {}


@functools.cache
def _capture_info_api():
    """The HIP capture ID distinguishes graphs built on the same stream (stdlib ctypes, no device launch)."""
    lib = ctypes.CDLL("libamdhip64.so")
    fn = lib.hipStreamGetCaptureInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulonglong)]
    fn.restype = ctypes.c_int
    return lib, fn


def _counter_key(device):
    stream = torch.cuda.current_stream(device).cuda_stream
    if not torch.cuda.is_current_stream_capturing():
        return device, ("stream", stream)
    _, fn = _capture_info_api()
    status, capture_id = ctypes.c_int(), ctypes.c_ulonglong()
    rc = fn(ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(capture_id))
    if rc or status.value != 1:
        raise RuntimeError(f"hipStreamGetCaptureInfo failed: rc={rc}, status={status.value}")
    return device, ("capture", capture_id.value, stream)


def _counters(device: torch.device) -> torch.Tensor:
    """One counter set per eager stream or (capture ID, stream), zeroed once outside capture: graphs captured on one
    stream may replay concurrently on others, and each call adds ``EPOCH`` to every counter it uses. Pool
    initialization and slice reservation are serialized across host threads (tensor ops may release the GIL)."""
    key = _counter_key(device)
    cnt = _COUNTERS.get(key)
    if cnt is not None:
        return cnt
    with _COUNTER_LOCK:
        cnt = _COUNTERS.get(key)
        if cnt is not None:
            return cnt
        pool = _POOL.get(device)
        if pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("mxfp8_qkv_norm_rope_nvfp4_insert: first call inside CUDA graph capture; call it "
                                   "once eagerly on the device first (the entry's warm-up does)")
            pool = torch.zeros(_COUNTER_SETS * _COUNTER_WORDS, dtype=torch.int32, device=device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            ready.synchronize()
            _POOL[device] = pool
        used = sum(1 for d, _ in _COUNTERS if d == device)
        if used >= _COUNTER_SETS:
            raise RuntimeError(f"mxfp8_qkv_norm_rope_nvfp4_insert: more than {_COUNTER_SETS} stream/capture "
                               f"counter sets on {device}")
        cnt = _COUNTERS[key] = pool[used * _COUNTER_WORDS:(used + 1) * _COUNTER_WORDS]
        return cnt


def _fused(activation, activation_scale, weight, weight_scale, m, q_norm_weight, k_norm_weight,
           index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
           kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out, eps):
    plan = _plan(m)
    slab = torch.empty((plan[3], m, N), dtype=torch.float32, device=activation.device)
    _run_compiled(
        _fused_launcher(plan),
        activation.data_ptr(), weight.data_ptr(), activation_scale.data_ptr(), weight_scale.data_ptr(),
        slab.data_ptr(), _counters(activation.device).data_ptr(), q_out.data_ptr(), index_q_out.data_ptr(),
        q_norm_weight.data_ptr(), k_norm_weight.data_ptr(), index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(), cos_sin_cache.data_ptr(), positions.data_ptr(), slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(), kv_cache.data_ptr(), index_cache.data_ptr(), k_scale.data_ptr(),
        v_scale.data_ptr(), m, activation_scale.numel() * activation_scale.element_size(), index_q_out.stride(0),
        kv_cache.stride(0), index_cache.stride(0), cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0], torch.cuda.current_stream(),
    )


def mxfp8_qkv_norm_rope_nvfp4_insert(
    activation: torch.Tensor,  # e4m3 codes in the (16, 16) layout, prequant_code_tile_rows(m, 4992, 6144) rows
    activation_scale: torch.Tensor,  # shuffled E8M0 scales [ceil(rows / 32) * 32, 192]
    weight: torch.Tensor,  # preshuffle_mxfp8_weight codes [4992, 6144]
    weight_scale: torch.Tensor,  # its shuffled scales
    m: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16
    positions: torch.Tensor,  # [m] int64
    slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 4, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [m, 4096] bf16
    index_q_out: torch.Tensor,  # [m, 256] e4m3, row stride free
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
) -> None:
    """``mxfp8_gemm_shuffled(defer_reduce=True)`` + the split-K reduce-insert (or the bf16 GEMM +
    ``norm_rope_nvfp4_insert``), bit for bit; one launch at 1-16 rows."""
    if m == 0:
        return
    assert (num_heads, num_kv_heads, num_index_heads) == (_kernel.NQ, _kernel.NKV, _kernel.NIQ)
    assert tuple(weight.shape) == (N, K) and activation.shape[1] == K
    assert cos_sin_cache.shape[1] == _kernel.ROT
    if not 1 <= m <= MAX_ROWS or not _gemm.prequant_code_tile_rows(m, N, K, torch.bfloat16):
        raise ValueError(f"no shuffled-code QKV row at M {m}")
    ins = (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
           slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out)
    if m <= FUSED_MAX_ROWS:
        assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
        assert activation.shape[0] >= 16 and activation.is_contiguous() and weight.is_contiguous()
        assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
        assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1
        assert index_q_out.stride(0) % 4 == 0
        assert kv_cache.shape[1:] == (2 * _kernel.NKV, 128, 72) and kv_cache.stride(1) == 128 * 72
        assert kv_cache.stride(3) == 1
        assert index_cache.shape[1:] == (128, 72) and index_cache.stride(1) == 72 and index_cache.stride(2) == 1
        _fused(activation, activation_scale, weight, weight_scale, m, *ins, eps)
        return
    kw = dict(num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps)
    config = _nt_config(m)
    if config is not None:
        assert activation.shape[0] == -(-m // config.tile_m) * config.tile_m
        qkv = torch.empty((m, N), dtype=torch.bfloat16, device=activation.device)
        n_mb = -(-m // config.tile_m)
        u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
        _run_compiled(_nt_gemm_launcher(config), u8(activation), u8(weight), u8(qkv), u8(activation_scale),
                      u8(weight_scale), m, activation_scale.numel() * activation_scale.element_size(), n_mb,
                      n_mb * (N // config.tile_n), torch.cuda.current_stream())
        norm_rope_nvfp4_insert(qkv, *ins, **kw)
        return
    qkv = _gemm.mxfp8_gemm_shuffled(activation, activation_scale, weight, weight_scale, torch.bfloat16, m,
                                    defer_reduce=True)
    if isinstance(qkv, _gemm.SkPartials):
        splitk_reduce_norm_rope_nvfp4_insert(qkv.tmp, qkv.split_k, *ins, **kw)
    else:
        norm_rope_nvfp4_insert(qkv, *ins, **kw)


def _mode(*args, **kwargs):
    """One mode: the variants depend on the token count only (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    """Every token count the entry serves (each picks its kernel by the count)."""
    return range(1, MAX_ROWS + 1)


def _inputs(mode, n, activation, activation_scale, weight, weight_scale, m, q_norm_weight, k_norm_weight,
            index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
            kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out, **kw):
    """One warm-up call at ``n`` tokens: zero codes in the producer's tile, every slot -1 (no page is written)."""
    dev = weight.device
    rows = _gemm.prequant_code_tile_rows(n, N, K, torch.bfloat16)
    rows = (n + rows - 1) // rows * rows
    codes = torch.zeros((rows, K), dtype=activation.dtype, device=dev)
    scales = torch.full(((rows + 31) // 32 * 32, K // 32), 127, dtype=activation_scale.dtype, device=dev)
    zeros = torch.zeros(n, dtype=torch.int64, device=dev)
    skip = torch.full((n,), -1, dtype=torch.int64, device=dev)
    q = torch.empty((n, q_out.shape[1]), dtype=q_out.dtype, device=dev)
    iq = torch.empty((n, index_q_out.shape[1]), dtype=index_q_out.dtype, device=dev)
    return ((codes, scales, weight, weight_scale, n, q_norm_weight, k_norm_weight, index_q_norm_weight,
             index_k_norm_weight, cos_sin_cache, zeros, skip, skip, kv_cache, index_cache, k_scale, v_scale, q, iq),
            kw)


mxfp8_qkv_norm_rope_nvfp4_insert = entry("attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128",
                                         mode=_mode, sizes=_sizes, inputs=_inputs)(mxfp8_qkv_norm_rope_nvfp4_insert)
