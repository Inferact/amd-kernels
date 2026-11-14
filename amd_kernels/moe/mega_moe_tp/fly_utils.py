# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Device helpers of the MegaMoE TP kernels and their launch helpers.

From AITER ``aiter/ops/flydsl/kernels/{kernels_common,act,mxfp4_gemm_common,communication_ops_utils,tensor_shim}.py``
(ROCm/aiter edf58c67a), only what this package uses.
"""

import os
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.utils import env

LOG2E = 1.4426950408889634


def ceildiv(numer, denom):
    """Ceiling division preserving Python-int or DSL-scalar operand types."""
    return (numer + denom - 1) // denom


def _to_raw(v):
    """ArithValue / Numeric (Int32, Boolean, ...) -> raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


# ---- activations ----
def sigmoid_batch(xs, *, alpha=1.0):
    """Emit all exponentials before their reciprocals to preserve batch scheduling."""
    e = [fx.exp2(x * fx.Float32(-alpha * LOG2E), fastmath="afn") for x in xs]
    return [fx.Float32(rocdl.rcp(T.f32, _to_raw(fx.Float32(1.0) + ei))) for ei in e]


def sigmoid_f32(g, *, alpha=1.0):
    return sigmoid_batch([g], alpha=alpha)[0]


def tanh_batch(xs):
    """Sign-restored tanh with exp2 and reciprocal operations grouped by stage."""
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    es = [fx.exp2(fx.max(x, -x) * neg_two_log2e, fastmath="afn") for x in xs]
    recips = [fx.Float32(rocdl.rcp(T.f32, _to_raw(fx.Float32(1.0) + e))) for e in es]
    zero = fx.Float32(0.0)
    out = []
    for i, x in enumerate(xs):
        tanh_abs = (fx.Float32(1.0) - es[i]) * recips[i]
        out.append((x > zero).select(tanh_abs, -tanh_abs))
    return out


def tanh_f32(x):
    return tanh_batch([x])[0]


def _silu_mul_batch(gs, us):
    sig = sigmoid_batch(gs)
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _clamp_gate_up_batch(gate_values, up_values, limit=7.0):
    limit_f32 = fx.Float32(float(limit))
    neg_limit_f32 = fx.Float32(-float(limit))
    gates = [fx.min(gate, limit_f32) for gate in gate_values]
    ups = [fx.max(fx.min(up, limit_f32), neg_limit_f32) for up in up_values]
    return gates, ups


def _swiglu_mul_batch(gate_values, up_values, limit):
    gates, ups = _clamp_gate_up_batch(gate_values, up_values, limit)
    # alpha folded into the exp2 constant (this kernel's rounding)
    sigmoids = sigmoid_batch(gates, alpha=1.702)
    return [gate * sigmoid * (up + 1.0) for gate, sigmoid, up in zip(gates, sigmoids, ups)]


def _situ_mul_batch(gate_values, up_values, beta=1.0, linear_beta=1.0):
    # unclamped, host-computed reciprocals, left-associated products
    beta_f32 = fx.Float32(float(beta))
    beta_rcp = fx.Float32(1.0 / float(beta))
    linear_beta_f32 = fx.Float32(float(linear_beta))
    linear_beta_rcp = fx.Float32(1.0 / float(linear_beta))
    return [
        beta_f32 * tanh_f32(gate * beta_rcp) * sigmoid_f32(gate) * linear_beta_f32 * tanh_f32(up * linear_beta_rcp)
        for gate, up in zip(gate_values, up_values)
    ]


def _activation_mul_batch(gate_values, up_values, act="silu", situ_beta=1.0, situ_linear_beta=1.0,
                          swiglu_limit=7.0):
    if act == "swiglu":
        return _swiglu_mul_batch(gate_values, up_values, limit=swiglu_limit)
    if act == "situv2":
        return _situ_mul_batch(gate_values, up_values, beta=situ_beta, linear_beta=situ_linear_beta)
    # SiLU clamps gate / up before the multiply; the host passes +inf for an unclamped SiLU
    gate_values, up_values = _clamp_gate_up_batch(gate_values, up_values, swiglu_limit)
    return _silu_mul_batch(gate_values, up_values)


# ---- MX scales ----
def _fabs_f32(x):
    return fmath.absf(x)


def _e8m0_roundup(amax_f32, max_norm=6.0):
    wi = (amax_f32 * fx.Float32(1.0 / float(max_norm))).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = fx.Uint32(bexp) < 254
    return lt.select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32, max_norm=6.0):
    e8m0 = _e8m0_roundup(amax_f32, max_norm=max_norm)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


# ---- global memory ordering ----
def _to_ptr_global(v, dtype=fx.Int32, alignment=4):
    ptr_type = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment)
    return fx.inttoptr(ptr_type, fx.Int64(v))


def load_i32_global_agent(addr_i64):
    """Volatile monotonic i32 load with agent-one-as visibility."""
    return fx.generic_load(_to_ptr_global(addr_i64), volatile=True, memory_order=fx.AtomicOrdering.Monotonic,
                           syncscope=fx.rocdl.SyncScope.AgentOneAs)


def fence_release(syncscope):
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=syncscope)


def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    dtype = fx.Int32 if isinstance(val, int) else fx.Int64 if fx.as_ir_value(val).type == T.i64 else fx.Int32
    ptr = _to_ptr_global(addr_i64, dtype, dtype.width // 8)
    return fx.atomic_add(ptr, dtype(val), syncscope=fx.rocdl.SyncScope.Agent)


# ---- launch ----
_PRELOAD_COMPILE_LOCK = threading.RLock()


def _compile_and_run(exe, *args):
    """``flyc.compile(exe, *args)``: compiles and runs, returns the ``CompiledFunction`` (None in compile-only)."""
    try:
        return flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _run_compiled(exe, *args):
    """First call: compile and run; later calls: dispatch the cached ``CompiledFunction``."""
    if env.compile.compile_only:
        # a callable warmed before a compile-only preload must not dispatch: keep it for the run
        _compile_and_run(exe, *args)
        return
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    exe._cf = _compile_and_run(exe, *args)


def _preload_compiled(exe, *args):
    """Compile (or load from the disk cache) without dispatching a kernel: ``JitFunction.preload`` where FlyDSL has
    it, else FlyDSL's ``COMPILE_ONLY`` mode around ``flyc.compile``."""
    preload = getattr(exe, "preload", None)
    if callable(preload):
        return preload(*args)
    with _PRELOAD_COMPILE_LOCK:
        old = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            return flyc.compile(exe, *args)
        finally:
            if old is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = old
