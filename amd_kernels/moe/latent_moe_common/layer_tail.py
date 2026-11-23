# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the decode layer tail in one persistent launch (o_proj all-reduce, AttnRes, router, shared and routed
experts, latent tail, final all-reduce), routed by (hidden, o_proj K, experts, world size). The runner is built per
key; its launches route by the runner."""
import dataclasses
import functools
from collections.abc import Mapping

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

from . import dispatch

# hidden, o_proj K (w_o [hidden, K]), routed experts (w13 [E, ...]), world
_NAMES = ("hidden", "o_proj_k", "experts", "world")
_ROUTES = Routes("layer_tail_launch", {(7168, 1536, 896, 8): "amd_kernels.moe.layer_tail_k2_h7168_ranks8.host"}, _NAMES)
_RUNNER = Routes("layer_tail_runner", _ROUTES.hosts, _NAMES)
_OP = _ROUTES.op
MAX_BLOCKS = 8  # the AttnRes block sources LayerTailRunner sizes its scratch for


@functools.cache
def _runner_host(cls):
    key = _ROUTES.owner(cls)
    if key is None:
        raise ValueError(dispatch.not_owned(_ROUTES, cls))
    host = _ROUTES.package(key)
    return host, frozenset(f.name for f in dataclasses.fields(host.K2Build))


# build keys that put another kernel's work into the launch (an MLA / DSA layer's M1b, M1c, DSA attention); the
# package fuses none of them, so a build that sets one is declined. Their other keys only parameterise these parts.
_FUSED_PARTS = ("m1b_qlen", "m1c", "dsa")


def _unbuilt(fields, items):
    return sorted(k for k, v in items if k in _FUSED_PARTS and k not in fields and v)


@functools.cache
def _k2_build(cls, items):
    host, fields = _runner_host(cls)
    extra = _unbuilt(fields, items)
    if extra:
        raise ValueError(f"{_OP}: the kernel does not fuse {extra}")
    return host.K2Build(**{k: v for k, v in items if k in fields})


def layer_tail_runner(device, group, rank, *, hidden, o_proj_k, experts, min_scratch=0):
    """The per-rank state of the launches (collective over ``group``): ``.step_begin()`` once a forward before its
    first launch, ``.close()``."""
    host = _RUNNER.package((hidden, o_proj_k, experts, dist.get_world_size(group)))
    return host.LayerTailRunner(device, group, rank, min_scratch=min_scratch)


def layer_tail_runner_blocker(device, group, rank, *, hidden, o_proj_k, experts, min_scratch=0) -> str | None:
    if not routes.available():
        return f"{_RUNNER.op}: needs gfx950 with FlyDSL"
    return _RUNNER.blocker((hidden, o_proj_k, experts, dist.get_world_size(group)))


def prepare_layer_tail_runner(device, group, rank, *, hidden, o_proj_k, experts, min_scratch=0):
    """The runner of this rank, or the reason none serves this key."""
    return prepare(layer_tail_runner_blocker(device, group, rank, hidden=hidden, o_proj_k=o_proj_k, experts=experts,
                                             min_scratch=min_scratch),
                   lambda: layer_tail_runner(device, group, rank, hidden=hidden, o_proj_k=o_proj_k, experts=experts,
                                             min_scratch=min_scratch))


def layer_tail_launch(runner, build: Mapping, *, core, w_o, prefix, blocks, ares_nw, ares_qk, in_nw, w_gate, bias,
                      w_ld, w_sgu, w_sd, w13, w13s, w2, w2s, ln_w, w_up, out, layer, tl=None):
    """One layer's tail: ``core`` [S, K] (o_proj's input) -> ``out`` [S, hidden]; ``prefix`` [S, hidden] updated in
    place. ``build`` maps the package's build fields (tokens, nblocks, eps, reset, ...) to hashable values; other keys
    are ignored, except that a build asking for a fused MLA / DSA part (``m1b_qlen``, ``m1c``, ``dsa``) is declined."""
    cls = type(runner)
    key = _k2_build(cls, tuple(build.items()))
    _runner_host(cls)[0].k2_launch(
        key, core=core, w_o=w_o, prefix=prefix, blocks=blocks, ares_nw=ares_nw, ares_qk=ares_qk, in_nw=in_nw,
        w_gate=w_gate, bias=bias, w_ld=w_ld, w_sgu=w_sgu, w_sd=w_sd, w13=w13, w13s=w13s, w2=w2, w2s=w2s, ln_w=ln_w,
        w_up=w_up, out=out, layer=layer, tl=tl, **runner.launch_args(), **runner.peer_args())
    return out


def layer_tail_launch_shapes() -> frozenset:
    return _ROUTES.shapes()


def layer_tail_launch_blocker(runner, build: Mapping, *, core, w_o, prefix, blocks, ares_nw, ares_qk, in_nw, w_gate,
                              bias, w_ld, w_sgu, w_sd, w13, w13s, w2, w2s, ln_w, w_up, out, layer,
                              tl=None) -> str | None:
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _ROUTES.owner(type(runner))
    if key is None:
        return dispatch.not_owned(_ROUTES, type(runner))
    host = _ROUTES.package(key)
    call = (prefix.shape[-1], w_o.shape[-1], w13.shape[0])
    if call != key[:3]:
        return dispatch.built_for(_ROUTES, key, call, "runner")
    if "tokens" not in build or "nblocks" not in build:
        return f"{_OP}: build needs tokens and nblocks"
    try:
        hash(tuple(build.items()))
    except TypeError:
        return f"{_OP}: build values must be hashable"
    extra = _unbuilt(_runner_host(type(runner))[1], build.items())
    if extra:
        return f"{_OP}: the kernel does not fuse {extra}"
    tokens, nblocks = build["tokens"], build["nblocks"]
    if not isinstance(tokens, int) or not isinstance(nblocks, int):
        return f"{_OP}: build tokens and nblocks must be int"
    if tokens not in host.SERVED_ROWS or tokens != core.shape[0]:
        return f"{_OP}: {tokens} tokens, core has {core.shape[0]} rows (the kernel takes {host.SERVED_ROWS})"
    if not 0 <= nblocks <= MAX_BLOCKS:
        return f"{_OP}: {nblocks} block sources (the runner holds 0..{MAX_BLOCKS})"
    hidden, dev = key[0], runner.scratch.device
    if tuple(prefix.shape) != (tokens, hidden) or tuple(out.shape) != (tokens, hidden):
        return f"{_OP}: prefix and out must be [{tokens}, {hidden}]"
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != dev for t in (core, prefix, out)):
        return f"{_OP}: core, prefix and out must be contiguous bf16 on the runner device"
    if (blocks.ndim != 3 or blocks.shape[0] != tokens or blocks.shape[1] < nblocks or blocks.shape[2] != hidden
            or blocks.dtype != torch.bfloat16 or blocks.stride(2) != 1 or blocks.device != dev):
        return f"{_OP}: blocks must be bf16 [{tokens}, >={nblocks}, {hidden}] with unit inner stride"
    reason = host.weights_unsupported_reason(dev, w_o=w_o, ares_nw=ares_nw, ares_qk=ares_qk, in_nw=in_nw, w_gate=w_gate,
                                             bias=bias, w_ld=w_ld, w_sgu=w_sgu, w_sd=w_sd, w13=w13, w13s=w13s, w2=w2,
                                             w2s=w2s, ln_w=ln_w, w_up=w_up)
    return None if reason is None else f"{_OP}: {reason}"


layer_tail_runner_unsupported_reason = renamed(layer_tail_runner_blocker, "layer_tail_runner_unsupported_reason")
layer_tail_launch_unsupported_reason = renamed(layer_tail_launch_blocker, "layer_tail_launch_unsupported_reason")
