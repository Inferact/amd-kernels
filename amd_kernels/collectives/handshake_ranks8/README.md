# Eight-rank hand-off (flag barrier and small-payload exchanges)

## What it does

Every rank of an 8-GPU group must learn that every peer's data has landed. This package holds the measured minimum-latency
protocol: an uncached block per rank mapped on all peers, atomic system-scope flag stores into the peer's block, polls of the
rank's own block, the call number (epoch) in the flag word (nothing is reset, the counter is never rewound), kernel-argument
peer bases, two parity slots. Ordering contract: payload stores, `s_waitcnt vmcnt(0)` (their acknowledgements), then the
flag; or no flag at all (epoch-tagged 8-byte words that the receiver polls). Protocol and hazards: module docstring of
`kernel.py`. Contract (`reference.py`): `out[s] = x_s[me]` over int32 words, bit for bit; the barrier returns once all
ranks entered.

## Entry points

- `make_comm`: the communicator (a collective: uncached IPC blocks, rotated peer bases, epoch counters), before capture.
- `barrier`: flag hand-off only (`wave`, `lane`, `pull`).
- `exchange`: all-to-all of an int32 [8, n] block, n in `SIZES`: `ll` (epoch-tagged words), `flag` (acknowledged data,
  then flag), `pull` (rows stay at the sender); `host.exchange` picks `ll` up to 512 words per peer, else `flag`.

The device helpers other eight-rank kernels call (`peer_lane`, `raise_flag`, `wait_flag`, `epoch_load`, `flag_store`,
`spin_until`, `sel8`) live in `common/rank8_signal.py`; the communicator in `common/rank8_comm.py`.

## Performance

8 x MI355X, graph replay, steady-state us per call = (512-call graph - 16-call graph) / 496, MAX over ranks, min of 21
replays (a replay's start skew across the 8 processes, 10-50 us, is a fixed cost that a short graph divides over its
calls and a differenced pair removes). Floor = empty kernel + 0.8 us flag one-way + the path's measured phases (`flag`:
acknowledgement wait 0.64 + landed-data read 0.85; `pull`: 0.28 + 0.15 + 0.85) + egress bytes / 343 GB/s.

| call (words per peer x workgroups) | egress KB | us | over empty | floor us | floor / measured |
|---|---:|---:|---:|---:|---:|
| empty kernel | 0 | 1.53 | +0.00 | | |
| barrier `wave` (1 wg, 8 waves) | 0 | 2.05 | +0.51 | 2.33 | 114% |
| barrier `pull` | 0 | 1.98 | +0.45 | 2.33 | 118% |
| barrier `lane` (1 wave) | 0 | 2.03 | +0.50 | 2.33 | 115% |
| barrier `lane`, 8 workgroups each | 0 | 2.66 | +1.12 | 2.33 | 88% |
| `ll` 64 | 2 | 2.38 | +0.85 | 2.34 | 98% |
| `ll` 128 | 4 | 2.36 | +0.82 | 2.34 | 99% |
| `ll` 256 | 7 | 2.64 | +1.11 | 2.35 | 89% |
| `ll` 512 | 14 | 3.52 | +1.98 | 2.37 | 67% |
| `ll` 4096 x 8 | 112 | 4.10 | +2.56 | 2.66 | 65% |
| `flag` 64 | 2 | 4.00 | +2.47 | 3.83 | 96% |
| `flag` 1024 | 28 | 4.64 | +3.10 | 3.90 | 84% |
| `flag` 4096 x 4 | 112 | 5.12 | +3.58 | 4.15 | 81% |
| `flag` 16384 x 16 | 448 | 6.04 | +4.50 | 5.16 | 85% |
| `pull` 64 | 2 | 3.91 | +2.38 | 3.62 | 92% |
| `pull` 1024 | 28 | 4.27 | +2.73 | 3.69 | 87% |
| `pull` 4096 | 112 | 7.83 | +6.29 | 3.94 | 50% |

Barrier floors above 100% beat the 0.8 us ping-pong one-way model (the next kernel's launch overlaps the wait). `ll` is
faster than `flag` up to 512 words on one workgroup and up to 4096 words (16 KB per peer) on 8; beyond that `flag` wins
(8192 x 8: 6.61 vs 5.39 us).

## Tests

```bash
python3 -m pytest tests/collectives/handshake_ranks8 -q                                          # 8 GPUs
AKR_COMPILE_ONLY=1 python3 -m pytest tests/collectives/handshake_ranks8/test_compile.py -q       # 1 GPU
python3 benchmarks/collectives/handshake_ranks8/benchmark_barrier.py empty,wave,ex_ll:64,ex_flag:1024   # 8 GPUs
```
