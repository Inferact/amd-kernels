# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MegaMoE TP on 4 ranks vs ``reference.mega_moe_tp_ref`` (rel L2 <= RTOL over every rank): both comm modes and
formats at decode / prefill token counts and routings, then the layer's contract: varying m, ``out=``, graph replay ==
eager on new inputs, both schedules, masked routes, m = 0, layers built and freed, the fused tail, host validation,
and the big-M path where MORI is importable."""

import gc

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import WORLD, run_on_4_gpus

from amd_kernels.moe.mega_moe_tp import mega_moe_tp_supported
from amd_kernels.moe.mega_moe_tp.reference import RTOL, tail_ref

from .workload import SEED, SHAPES, Case, build_weights, make_case, new_layer, rank_max, rel_l2, same_on_all_ranks

MAX_LOCAL = 256  # max_local_tokens of the layers: 1024 tokens on 4 ranks
_WEIGHTS = {}


@pytest.fixture(autouse=True)
def _gfx950():
    if not torch.cuda.is_available() or not mega_moe_tp_supported():
        pytest.skip("needs gfx950")


def _weights(name, rank):
    if name not in _WEIGHTS:
        _WEIGHTS.clear()
        gc.collect()
        _WEIGHTS[name] = build_weights(SHAPES[name], rank, WORLD, torch.device("cuda", rank))
    return SHAPES[name], _WEIGHTS[name]


def _prepare(layer, tokens, **kw):
    """the layer's contract: every launch config armed and run once (a watchdog timeout reset and retried) before
    use; a first launch that loads its kernel can keep a rank past its peers' wait"""
    ar = layer.cfg.comm_mode == "ar"
    layer.prepare(sorted({-(-t // WORLD) if ar else t // WORLD for t in tokens}), **kw)


def _call(layer, c: Case, errors, what, out=None):
    layer.clear_errors()
    y = layer(c.x, c.w, c.ids, out=out)
    torch.cuda.synchronize()
    if rank_max(layer.poll_errors()):
        errors.append(f"{what}: watchdog error")
    if rank_max(float(torch.isnan(y).any())):
        errors.append(f"{what}: NaN in the output")
    return y


def _check(y, c: Case, errors, what):
    e = rel_l2(y, c.ref)
    if not e <= RTOL:
        errors.append(f"{what}: rel_l2 {e:.4f} > {RTOL}")


def _expect_raise(fn, errors, what):
    raised = 0.0
    try:
        fn()
        torch.cuda.synchronize()
    except (ValueError, TypeError):
        raised = 1.0
    if -rank_max(-raised) < 1.0:
        errors.append(f"{what} was accepted (expected a host-side error)")


def _accuracy(rank, errors, name, mode, comm_dtype, ar_gather, tokens, routes):
    shape, wt = _weights(name, rank)
    layer = new_layer(shape, wt, rank, WORLD, mode, MAX_LOCAL, comm_dtype, ar_gather)
    _prepare(layer, tokens)
    for t in tokens:
        for i, kind in enumerate(routes):
            c = make_case(shape, wt, rank, WORLD, mode, t, kind, SEED + 1000 * i + t)
            out = torch.empty_like(c.ref, dtype=torch.bfloat16)
            what = f"{name} {mode} {comm_dtype}/{ar_gather} M={t} {kind}"
            y = _call(layer, c, errors, what, out=out)
            _check(y, c, errors, what)
            if mode == "ar" and not same_on_all_ranks(y):
                errors.append(f"{what}: the all-reduced output differs across ranks")
    layer.close()


@pytest.mark.parametrize(("mode", "comm_dtype", "ar_gather"), [
    ("ag_rs", "fp8", "bf16"), ("ag_rs", "bf16", "bf16"), ("ar", "fp8", "bf16"), ("ar", "bf16", "bf16"),
    ("ar", "fp8", "fp8"),
])  # fmt: skip
def test_mega_moe_tp_accuracy(mode, comm_dtype, ar_gather):
    run_on_4_gpus(_accuracy, "m3", mode, comm_dtype, ar_gather, (8, 64, 256, 1024), ("balanced", "random", "hot"))


@pytest.mark.parametrize("mode", ["ag_rs", "ar"])
def test_mega_moe_tp_accuracy_glm5(mode):
    run_on_4_gpus(_accuracy, "glm5", mode, "fp8", "bf16", (8, 256), ("random",))


def _features(rank, errors, mode):
    shape, wt = _weights("m3", rank)
    world = WORLD
    layer = new_layer(shape, wt, rank, world, mode, MAX_LOCAL)
    _prepare(layer, [m * world for m in (1, 4, 8, 16, 32, 64)])
    for m in (8, 1, 32, 1, 64, 8):  # varying m on one layer
        c = make_case(shape, wt, rank, world, mode, m * world, "random", SEED + m)
        _check(_call(layer, c, errors, f"m={m}").clone(), c, errors, f"m={m}")

    c1 = make_case(shape, wt, rank, world, mode, 16 * world, "random", 1)
    c2 = make_case(shape, wt, rank, world, mode, 16 * world, "random", 2)
    out = torch.empty_like(c1.ref, dtype=torch.bfloat16)
    if _call(layer, c1, errors, "out=", out=out).data_ptr() != out.data_ptr():
        errors.append("out= was not returned")
    keep = out.clone()
    _call(layer, c2, errors, "after out=")
    if rank_max(float(not torch.equal(out, keep))):
        errors.append("out= of call N changed by call N+1")
    _check(keep, c1, errors, "out=")

    c = make_case(shape, wt, rank, world, mode, 16 * world, "random", 11, mask=0.2)
    _check(_call(layer, c, errors, "masked").clone(), c, errors, "masked ids")
    c = make_case(shape, wt, rank, world, mode, 16 * world, "random", 12)
    _check(_call(layer, c, errors, "after masked").clone(), c, errors, "the call after masked ids")

    c = make_case(shape, wt, rank, world, mode, 4 * world, "random", 13)
    y = _call(layer, Case(c.x[:0], c.w[:0], c.ids[:0], c.ref[:0], c.ids_all), errors, "m=0")
    if tuple(y.shape) != (0, shape.model_dim):
        errors.append(f"m=0 returned shape {tuple(y.shape)}")
    _check(_call(layer, c, errors, "after m=0").clone(), c, errors, "the call after m=0")

    # graph replay on new inputs == eager (stale peer data would show)
    cases = [make_case(shape, wt, rank, world, mode, 16 * world, "random", s) for s in (3, 4, 5)]
    eager = [_call(layer, k, errors, "eager").clone() for k in cases]
    x, w, ids = cases[0].x.clone(), cases[0].w.clone(), cases[0].ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(x, w, ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = layer(x, w, ids)
    same = True
    for i in range(12):
        k = i % len(cases)
        for dst, src in ((x, cases[k].x), (w, cases[k].w), (ids, cases[k].ids)):
            dst.copy_(src)
        g.replay()
        same = same and torch.equal(y, eager[k])
    torch.cuda.synchronize()
    if rank_max(layer.poll_errors()):
        errors.append("watchdog error during graph replay")
    if rank_max(float(not same)):
        errors.append("graph replay != eager")
    del g
    layer.close()

    # layers built and freed between calls
    layers = [new_layer(shape, wt, rank, world, mode, MAX_LOCAL) for _ in range(3)]
    for rnd in range(2):
        for lay in layers:
            _prepare(lay, [8 * world])
        for i, lay in enumerate(layers):
            c = make_case(shape, wt, rank, world, mode, 8 * world, "random", 100 * rnd + i)
            _check(_call(lay, c, errors, f"layer {i}").clone(), c, errors, f"round {rnd} layer {i}")
        for lay in layers[:2]:
            lay.close()
        del layers[:2]
        gc.collect()
        layers += [new_layer(shape, wt, rank, world, mode, MAX_LOCAL) for _ in range(2)]
    for lay in layers:
        lay.close()


@pytest.mark.parametrize("mode", ["ag_rs", "ar"])
def test_mega_moe_tp_features(mode):
    run_on_4_gpus(_features, mode)


def _schedules(rank, errors, mode, lb):
    """the dynamic schedule up to 256 tokens; lb: the large-batch one from 1 token up at its smallest row tile (the
    mixed GEMM2 split); odd local token counts and a narrow routing"""
    shape, wt = _weights("m3", rank)
    kw = {"lb_min": 1, "lb_mt": 3, "lb_mt_small": 3, "lb_npp": 2, "lb_q": 3} if lb else {"lb_min": 0}
    layer = new_layer(shape, wt, rank, WORLD, mode, MAX_LOCAL, **kw)
    _prepare(layer, (WORLD, 3 * WORLD, 33 * WORLD, MAX_LOCAL * WORLD))
    for t in (WORLD, 3 * WORLD, 33 * WORLD, MAX_LOCAL * WORLD):
        for kind in ("random", "subset"):
            c = make_case(shape, wt, rank, WORLD, mode, t, kind, SEED + 3 * t)
            _check(_call(layer, c, errors, f"M={t} {kind}").clone(), c, errors, f"lb={lb} M={t} {kind}")
    layer.close()


@pytest.mark.parametrize("lb", [False, True])
@pytest.mark.parametrize("mode", ["ag_rs", "ar"])
def test_mega_moe_tp_schedules(mode, lb):
    run_on_4_gpus(_schedules, mode, lb)


def _tail(rank, errors):
    """the fused tail in place, as a caller runs it: tail=(res, res, w)"""
    shape, wt = _weights("m3", rank)
    layer = new_layer(shape, wt, rank, WORLD, "ag_rs", MAX_LOCAL, lb_min=1)
    _prepare(layer, [MAX_LOCAL * WORLD])
    eng = layer.engine
    t = MAX_LOCAL * WORLD
    if not layer.tail_ok(t // WORLD):
        errors.append(f"no fused tail at {t} tokens")
        layer.close()
        return
    c = make_case(shape, wt, rank, WORLD, "ag_rs", t, "random", 21)
    dev = c.x.device
    res = torch.randn(c.x.shape, generator=torch.Generator(device=dev).manual_seed(77 + rank), device=dev)
    res = res.to(torch.bfloat16)
    nw = (0.1 * torch.randn((shape.model_dim,), generator=torch.Generator(device=dev).manual_seed(78), device=dev))
    nw = nw.to(torch.bfloat16)
    y0 = _call(layer, c, errors, "tail: plain").clone()
    r_ref, xn, s_ref = tail_ref(y0, res, nw, eng.tn_eps)
    # every rank's normed rows and scales (rank r's at r * m)
    xn_all, s_all = [torch.zeros((WORLD,) + v.shape) for v in (xn, s_ref)]
    xn_all[rank], s_all[rank] = xn.cpu(), s_ref.cpu()
    dist.all_reduce(xn_all)
    dist.all_reduce(s_all)
    xn, s_ref = xn_all.flatten(0, 1).to(dev), s_all.flatten(0, 1).to(dev)
    fp8 = torch.float8_e4m3fn
    q_ref = (xn.to(torch.bfloat16).float() / s_ref[:, None]).clamp(-448, 448).to(fp8)
    m = c.x.shape[0]
    own = slice(rank * m, (rank + 1) * m)
    # tn_local (the default): only this rank's rows of q / scales / bf16 rows are written; False: every rank's
    for local in (True, False):
        eng.tn_local = local
        _prepare(layer, [t], tail=True, tail_bf16=True)
        rows = own if local else slice(None)
        xn_r, s_r = xn[rows], s_ref[rows]
        e_ref = float((q_ref[rows].float() * s_r[:, None] - xn_r).norm() / xn_r.norm())
        for bf16 in (False, True):
            r = res.clone()
            outs = layer(c.x, c.w, c.ids, tail=(r, r, nw), bf16=bf16)
            torch.cuda.synchronize()
            y, q, s = outs[:3]
            shapes_ok = q.shape[0] == s.shape[0] == m * WORLD and (not bf16 or outs[3].shape[0] == m * WORLD)
            q, s = q[rows], s[rows]
            e_q = float((q.view(fp8).float() * s[:, None] - xn_r).norm() / xn_r.norm())
            e_rows = float((outs[3][rows].float() - xn_r).norm() / xn_r.norm()) if bf16 else 0.0
            bad = [what for what, ok in (
                ("watchdog", not layer.poll_errors()),
                ("shapes [T, ...]", shapes_ok),
                ("y", torch.equal(y, y0)),
                ("res_out", torch.equal(r, r_ref)),
                (f"q rel {e_q:.4f} (fp8 {e_ref:.4f})", e_q < 1.05 * e_ref),
                ("scale", float(((s - s_r).abs() / s_r).max()) < 1e-2),
                (f"bf16 rows rel {e_rows:.4f}", e_rows < 0.01),
            ) if not ok]  # fmt: skip
            if bad:
                errors.append(f"rank {rank} tail tn_local={local} bf16={bf16}: {bad}")
    eng.tn_local = True
    layer.close()


def test_mega_moe_tp_fused_tail():
    run_on_4_gpus(_tail)


def _validation(rank, errors):
    import os

    shape, wt = _weights("m3", rank)
    _expect_raise(lambda: new_layer(shape, wt, rank, WORLD, "ag_rs", 4096, bigm_min_tokens=0), errors,
                  "max_local_tokens=4096 without the big-M path")
    layer = new_layer(shape, wt, rank, WORLD, "ag_rs", MAX_LOCAL)
    _prepare(layer, [8 * WORLD])
    c = make_case(shape, wt, rank, WORLD, "ag_rs", 8 * WORLD, "random", 5)
    x, w, ids = c.x, c.w, c.ids
    _expect_raise(lambda: layer(x.float(), w, ids), errors, "fp32 hidden states")
    _expect_raise(lambda: layer(x[:, : shape.model_dim // 2].contiguous(), w, ids), errors, "hidden size")
    _expect_raise(lambda: layer(x, w[:, :-1].contiguous(), ids[:, :-1].contiguous()), errors, "topk")
    _expect_raise(lambda: layer(x, w, ids.to(torch.int16)), errors, "int16 topk_ids")
    _expect_raise(lambda: layer(x, w, ids.cpu()), errors, "topk_ids on the cpu")
    _expect_raise(lambda: layer(x, w, ids, out=torch.empty_like(x[:1])), errors, "out of the wrong shape")
    layer.close()
    # AKR_MEGA_MOE_TP_CHECK=1: the cross-rank input checks
    os.environ["AKR_MEGA_MOE_TP_CHECK"] = "1"
    try:
        layer = new_layer(shape, wt, rank, WORLD, "ag_rs", MAX_LOCAL)
        m = 16 if rank == 0 else 8
        c = make_case(shape, wt, rank, WORLD, "ag_rs", 16 * WORLD, "random", 14)
        _expect_raise(lambda: layer(c.x[:m], c.w[:m], c.ids[:m]), errors, "ag_rs with uneven m")
        layer.close()
        layer = new_layer(shape, wt, rank, WORLD, "ar", MAX_LOCAL)
        _prepare(layer, [8 * WORLD])
        c = make_case(shape, wt, rank, WORLD, "ar", 8 * WORLD, "random", 15)
        ids = c.ids.clone()
        if rank == 1:
            ids[0, 0] = (ids[0, 0] + 1) % shape.experts
        _expect_raise(lambda: layer(c.x, c.w, ids), errors, "ar with routing that differs across ranks")
        _check(_call(layer, c, errors, "checked").clone(), c, errors, "checked call")
        layer.close()
    finally:
        os.environ.pop("AKR_MEGA_MOE_TP_CHECK", None)


def test_mega_moe_tp_validation():
    run_on_4_gpus(_validation)


def _mori_importable() -> bool:
    from amd_kernels.moe.mega_moe_tp import sdma

    try:
        sdma._mori()
        return True
    except ImportError:
        return False


def _big_m(rank, errors, mode):
    pad = 64
    shape, wt = _weights("m3", rank)
    layer = new_layer(shape, wt, rank, WORLD, mode, MAX_LOCAL, bigm_min_tokens=pad * WORLD)
    if not layer.bigm_m:
        errors.append("no big-M path")
        return
    tokens = [(pad - 1) * WORLD, pad * WORLD, 4 * pad * WORLD] + ([pad * WORLD + 1] if mode == "ar" else [])
    _prepare(layer, tokens)
    for t in tokens:
        c = make_case(shape, wt, rank, WORLD, mode, t, "random", SEED + t)
        out = torch.empty_like(c.ref, dtype=torch.bfloat16)
        if _call(layer, c, errors, f"big-M M={t}", out=out).data_ptr() != out.data_ptr():
            errors.append(f"big-M M={t}: out= was not returned")
        _check(out, c, errors, f"big-M M={t}")
    c = make_case(shape, wt, rank, WORLD, mode, pad * WORLD, "random", SEED)
    _expect_raise(lambda: layer(c.x, c.w, c.ids, out=torch.empty_like(c.ref, dtype=torch.float16)), errors,
                  "big-M out of the wrong dtype")
    layer.close()
    layer.close()


@pytest.mark.parametrize("mode", ["ag_rs", "ar"])
def test_mega_moe_tp_big_m(mode):
    if not _mori_importable():
        pytest.skip("needs MORI built with SDMA (importable or in AKR_MORI_PATH)")
    run_on_4_gpus(_big_m, mode)
