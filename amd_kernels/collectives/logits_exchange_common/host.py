# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""lm_head shard GEMM + exchange of its logits rows: routes by (vocab shard N, hidden K, world) to the exchange
package's host (``RowAllToAll``, ``BUFFERS``, ``RECV_ROWS``) and the GEMM package's host (its op and ``prepare``), and
composes the two in ``LogitsExchange``, the communicator the factory returns (each package imports only itself). The
ops are its methods (``gemm_exchange``, ``recv``, ``warmup``, ``close``); the reason of a ``gemm_exchange`` call routes
by the communicator's key."""
import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

OP = "logits_gemm_exchange"
_DIMS = ("n_shard", "k", "world")
_ROUTES = Routes("logits_exchange_comm", {
    (20480, 7168, 8): "amd_kernels.collectives.all_to_all_bf16_n20480_ranks8.host",
}, _DIMS)
# (GEMM host, op, rows up to which the GEMM pushes its rows itself in owner / all-gather mode): the crossovers measured
# in the exchange package's README (above them a local write and one contiguous push)
_GEMMS = {
    (20480, 7168, 8): ("amd_kernels.gemm.bf16_n20480_k7168.host", "bf16_gemm_n20480_k7168", {True: 96, False: 16}),
}
_GEMM_ROUTES = Routes(OP, {key: gemm[0] for key, gemm in _GEMMS.items()}, _DIMS)


class LogitsExchange:
    """The communicator of one (N, K, world): the exchange package's ``RowAllToAll`` (``exchange``) and the GEMM
    package's lm_head shard GEMM, with the GEMM's local output rows; built by ``logits_exchange_comm``."""

    def __init__(self, key: tuple, exchange, gemm_host, op: str, serial_above: dict):
        self.key, self.exchange = key, exchange
        self.device, self.max_rows, self.buffers = exchange.counter.device, exchange.max_rows, exchange.buffers
        self._gemm, self._serial_above = getattr(gemm_host, op), serial_above
        self._local = torch.empty(self.max_rows, key[0], dtype=torch.bfloat16, device=self.device)
        gemm_host.prepare(self.device, self.max_rows)  # the split-K workspace, not inside a capture

    @property
    def closed(self) -> bool:
        return self.exchange.closed

    def gemm_exchange(self, x, w, row_indices, counts=None, buf=0, serial_above=None):
        """The shard GEMM of the rows ``x[row_indices]`` and the exchange of its output (``counts``: owner mode, rows
        sorted by owner; None: every rank receives all rows) into receive buffer ``buf``: the GEMM pushes the rows
        itself up to ``serial_above`` rows (one launch less), above that it writes them locally and one contiguous push
        moves them (full-row streams suit the xGMI better than the tile's row segments). ``serial_above`` defaults to
        the measured crossover of the mode. Read the rows with ``recv``; returns the communicator."""
        assert not self.closed and row_indices.ndim == 1
        rows = row_indices.numel()
        if serial_above is None:
            serial_above = self._serial_above[counts is not None]
        if rows <= serial_above:
            self._gemm(x, w, row_indices=row_indices, push=(self.exchange.send_args(buf), counts))
            self.exchange.wait()
            return self
        out = self._local[:rows]
        self._gemm(x, w, out, row_indices=row_indices)
        self.exchange(out, counts, buf)
        return self

    def recv(self, rows: int, buf: int = 0) -> torch.Tensor:
        """This rank's receive buffer ``buf`` [rows, vocabulary] bf16, in place: valid until the next exchange into
        it."""
        return self.exchange.recv(rows, buf)

    def warmup(self) -> None:
        """Collective before capture: compiles the exchange's pushes and wait (the GEMM's are ``amd_kernels.warm_up``
        entries)."""
        self.exchange.warmup()

    def close(self) -> None:
        """Collective, after every graph that uses the communicator is retired."""
        self.exchange.close()


def logits_gemm_exchange_shapes() -> frozenset[tuple[int, int, int]]:
    """The covered (vocab shard N, K, world) of the lm_head shard [N, K]."""
    return _ROUTES.shapes()


def logits_exchange_comm(rank: int, group, device, *, n_shard: int, k: int, max_rows: int, buffers: int = 2):
    """The routed packages' ``LogitsExchange`` (collective on the group) with its first-use buffers allocated, so that
    no call allocates (graph capture). ``buffers=1``: buffer 1 is buffer 0, for a caller whose ranks all read an
    exchange's rows before any of them issues the next."""
    key = (n_shard, k, dist.get_world_size(group))
    exchange = _ROUTES.package(key).RowAllToAll(rank, group, device, max_rows=max_rows, buffers=buffers)
    _, op, serial_above = _GEMMS[key]
    return LogitsExchange(key, exchange, _GEMM_ROUTES.host(key), op, serial_above)


def logits_exchange_comm_blocker(rank: int, group, device, *, n_shard: int, k: int, max_rows: int,
                                 buffers: int = 2) -> str | None:
    """None if ``logits_exchange_comm`` with these arguments builds a communicator, else why not."""
    op = _ROUTES.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = (n_shard, k, dist.get_world_size(group))
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    host = _ROUTES.host(key)
    if not 1 <= max_rows <= host.RECV_ROWS:
        return f"{op}: max_rows {max_rows} (the kernel takes 1..{host.RECV_ROWS})"
    if not 1 <= buffers <= host.BUFFERS:
        return f"{op}: {buffers} buffers (the kernel takes 1..{host.BUFFERS})"
    if dist.get_rank(group) != rank:
        return f"{op}: rank {rank} is not this process's rank {dist.get_rank(group)} in the group"
    return None


def prepare_logits_exchange_comm(rank: int, group, device, *, n_shard: int, k: int, max_rows: int, buffers: int = 2):
    """``logits_exchange_comm(...)``, or the reason it cannot be built."""
    return prepare(logits_exchange_comm_blocker(rank, group, device, n_shard=n_shard, k=k, max_rows=max_rows,
                                                buffers=buffers),
                   lambda: logits_exchange_comm(rank, group, device, n_shard=n_shard, k=k, max_rows=max_rows,
                                                buffers=buffers))


def logits_gemm_exchange_blocker(comm, x: torch.Tensor, w: torch.Tensor, row_indices: torch.Tensor,
                                 counts: torch.Tensor | None = None, buf: int = 0,
                                 serial_above: int | None = None) -> str | None:
    """None if ``comm.gemm_exchange`` with these arguments is served, else why not (host checks only)."""
    if not routes.available():
        return f"{OP}: needs gfx950 with FlyDSL"
    key = comm.key if isinstance(comm, LogitsExchange) else None
    if key not in _ROUTES.hosts:
        return f"{OP}: no kernel for communicator {type(comm).__qualname__} (covered: {sorted(_ROUTES.hosts)})"
    n_shard, k, world = key
    if tuple(w.shape) != (n_shard, k):
        return f"{OP}: communicator built for {key}, call is {(*w.shape, world)}"
    if comm.closed:
        return f"{OP}: communicator is closed"
    dev = comm.device
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.device != dev:
        return f"{OP}: w must be contiguous bf16 on the communicator's device"
    if (x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != dev
            or x.numel() * x.element_size() > 0xFFFFFFFF):
        return f"{OP}: x must be contiguous bf16 [rows, {k}] (< 4 GiB) on the communicator's device"
    if row_indices.ndim != 1 or row_indices.dtype != torch.int64 or not row_indices.is_contiguous() \
            or row_indices.device != dev:
        return f"{OP}: row_indices must be a contiguous int64 vector on the communicator's device"
    rows = row_indices.numel()
    if not 1 <= rows <= comm.max_rows:
        return f"{OP}: {rows} rows (the communicator takes 1..{comm.max_rows})"
    if counts is not None and (tuple(counts.shape) != (world,) or counts.dtype != torch.int32
                               or not counts.is_contiguous() or counts.device != dev):
        return f"{OP}: counts must be contiguous int32 [{world}] on the communicator's device"
    buffers = _ROUTES.host(key).BUFFERS
    if not 0 <= buf < buffers:
        return f"{OP}: buffer {buf} (the communicator has {buffers})"
    return None


logits_exchange_comm_unsupported_reason = renamed(logits_exchange_comm_blocker,
                                                  "logits_exchange_comm_unsupported_reason")
logits_gemm_exchange_unsupported_reason = renamed(logits_gemm_exchange_blocker,
                                                  "logits_gemm_exchange_unsupported_reason")
