#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The kernel-test pipeline's three steps (Python standard library only).

    ci.py plan --mode changed|all [--base REV]   # the targets to run, as TSV: id, GPUs, command
    ci.py run --targets T --tree DIR --out DIR --durations FILE   # on the GPU node: pack the targets onto its GPUs
    ci.py report --out DIR [--durations FILE]    # Markdown summary on stdout; exit 1 if any target failed

plan: a target is one test file of one entry (`pytest tests/<entry>/test_x.py` on 1 GPU, or what its row or its
entry's row in tools/ci/targets.tsv says), plus `repo` (the top-level tests/test_*.py). One process per file is the
parallelism: a FlyDSL compile holds the GIL from tracing to the code object, so threads in one process compile one
kernel at a time. `changed` keeps the entries whose amd_kernels/, tests/ or benchmarks/ files the change touches
(README edits do not count); a change anywhere else (conftest.py, env.sh, tools/, pyproject.toml, this pipeline)
runs every entry. run: every target gets its own copy of the tree on the node-local disk with empty JIT caches
(nothing compiled by an earlier run is reused) and its GPUs through ROCR_VISIBLE_DEVICES. A card runs one target at
a time, so a target's failures and timings are its own (two per card give intermittent failures that do not
reproduce alone, and timings that measure the neighbour). Targets longer than LONG_S (by the last recorded
durations) start first, longest first; then the widest, then the longest. The exception: compile targets (GPUs
`compile`, compile-only mode: tests/compile_only.py) launch no FlyDSL kernel and time nothing, so up to COMPILE_LANES
of them share one card, held from the first one's start to the last one's end. A target with a precompile column
first runs that command (`same`: its own) in compile-only mode on the shared card, in its own tree: the target then
starts with this run's compiled kernels in its cache (same tree, same sources). The shared card is taken in the
first round; precompiles go before the compile targets, longest target first.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINUTES = 10  # a target without a recorded duration
TARGET_TIMEOUT_S = 75 * 60
LONG_S = 150  # targets longer than this start before all others
COMPILE_LANES = 8  # compile targets on their shared card at a time (CPU-bound: one compiling process each)
PRECOMPILE_TIMEOUT_S = 15 * 60  # compile-only runs no kernel: a test can wait on a flag forever; what it compiled stays


def entries(root):
    return sorted(
        f"{p.parent.name}/{p.name}"
        for p in (root / "tests").glob("*/*")
        if p.is_dir() and any(p.glob("test_*.py"))
    )


def manifest(root):
    """targets.tsv rows: {path: (GPUs or None for skip, command or skip reason or "", precompile or "")}."""
    rows = {}
    for line in (root / "tools/ci/targets.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, gpus, *rest = line.split("\t")
        count = None if gpus == "skip" else 0 if gpus == "compile" else int(gpus)  # compile: 0 cards of its own
        rows[path] = (count, rest[0] if rest else "", rest[1] if len(rest) > 1 else "")
    return rows


def entry_targets(root, entry, rows):
    """(id, GPUs or None, command or skip reason, precompile) of one entry: one per test file, or one for the entry
    when its row names a command."""
    gpus, command, pre = rows.get(entry, (1, "", ""))
    if gpus is None or command:
        return [(entry, gpus, command, pre)]
    out = []
    for f in sorted((root / "tests" / entry).glob("test_*.py")):
        path = f"{entry}/{f.name}"
        g, c, p = rows.get(path, (gpus, "", pre))
        out.append((path.removesuffix(".py"), g, c or f"python3 -m pytest tests/{path} -q", p))
    return out


def changed_entries(root, base):
    files = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", base, "HEAD"], check=True, capture_output=True, text=True
    ).stdout.split()
    picked = set()
    for f in files:
        if f.endswith(".md"):
            continue
        m = re.match(r"(amd_kernels|tests|benchmarks)/([^/]+)/([^/]+)/", f)
        if m and (root / "tests" / m[2] / m[3]).is_dir():
            picked.add(f"{m[2]}/{m[3]}")
        elif m and m[1] == "benchmarks":
            continue  # a benchmark without tests
        elif f.startswith(".github/") and not f.startswith(".github/workflows/kernel-tests"):
            continue  # other workflows do not run the kernels
        else:
            return None  # a shared file: everything
    return picked


def plan(a):
    every = entries(ROOT)
    if a.mode == "all":
        picked = set(every)
    else:
        picked = changed_entries(ROOT, a.base)
        picked = set(every) if picked is None else picked
    rows = manifest(ROOT)
    unknown = [p for p in rows if not (p in every or (p.endswith(".py") and (ROOT / "tests" / p).is_file()))]
    if unknown:
        sys.exit(f"tools/ci/targets.tsv names no entry or test file: {unknown}")
    print("repo\t1\tpython3 -m pytest tests/test_*.py -q\t")
    for entry in every:
        if entry not in picked:
            continue
        for tid, gpus, command, pre in entry_targets(ROOT, entry, rows):
            if gpus is None:
                print(f"skip {tid}: {command}", file=sys.stderr)
                continue
            print(f"{tid}\t{gpus}\t{command}\t{pre}")


def slug(target_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", target_id)


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = []
    for line in Path(a.targets).read_text().splitlines():
        tid, gpus, command, *rest = line.split("\t")
        targets.append(dict(id=tid, gpus=int(gpus), command=command, precompile=rest[0] if rest else ""))
    cards = os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    cards = [c for c in cards.split(",") if c] or [str(i) for i in range(8)]
    durations = json.loads(Path(a.durations).read_text()) if Path(a.durations).is_file() else {}
    for t in targets:
        if t["gpus"] > len(cards):
            sys.exit(f"{t['id']} needs {t['gpus']} GPUs, the job has {len(cards)}")
        t["est"] = durations.get(t["id"], DEFAULT_MINUTES * 60)
    # the shared card's queue: the precompiles first (each holds its target back), then the compile targets
    lane = sorted((dict(id=t["id"], of=t, gpus=0, est=t["est"]) for t in targets if t["precompile"]),
                  key=lambda j: -j["est"])
    lane += sorted((t for t in targets if not t["gpus"]), key=lambda t: -t["est"])
    # the run ends with its longest target: with the widest first, the four 4-GPU targets take every card and the
    # two ~210 s compile tests start 30 s late
    pending = sorted((t for t in targets if t["gpus"]), key=lambda t: (t["est"] <= LONG_S, -t["gpus"], -t["est"]))
    waiting = {t["id"] for t in targets if t["precompile"]}  # targets whose precompile has not ended
    trees = Path(a.trees)
    base = Path(a.tree)
    env0 = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}

    # the image and the container, once, before any target starts (concurrent first starts would race)
    env = dict(env0, ROCR_VISIBLE_DEVICES=",".join(cards))
    check = f"import torch; n = torch.cuda.device_count(); print('visible GPUs', n); assert n == {len(cards)}, n"
    if subprocess.run([str(base / "tools/enroot_test.sh"), "python3", "-c", check], env=env).returncode:
        (out / "summary.json").write_text(json.dumps(dict(error="GPU visibility check failed")))
        sys.exit(97)

    def start(job, job_cards):
        """Start a target (job is the target) or its precompile (job["of"] is the target) on job_cards."""
        t = job.get("of", job)
        tree = trees / slug(t["id"])
        if not t.get("tree"):
            tree.mkdir(parents=True, exist_ok=True)
            # no JIT cache survives from an earlier run, so every target compiles what it tests (--delete drops the
            # old .cache); its precompile fills that tree's cache in this run, from the same sources, and the target
            # reuses it
            subprocess.run(["rsync", "-a", "--delete", f"{base}/", f"{tree}/"], check=True)
            t["tree"] = str(tree)
        pre = "of" in job
        command = (t["command"] if t["precompile"] == "same" else t["precompile"]) if pre else t["command"]
        if command.startswith("python3 -m pytest"):
            command += " -p no:cacheprovider"
            if pre:
                # its tests fail (outputs unwritten): skip the failure reports, ~5.5 s each in a MoE test (tensor
                # reprs; 6 failing cases 35.7 -> 2.8 s on a warm cache)
                command += " --tb=no --assert=plain"
            else:
                command += f" --junitxml={shlex.quote(str(out / (slug(t['id']) + '.xml')))}"
        if pre or not t["gpus"]:
            command = f"AKR_COMPILE_ONLY=1 {command}"
        log = open(out / (slug(t["id"]) + (".precompile.log" if pre else ".log")), "w")
        log.write(f"$ ROCR_VISIBLE_DEVICES={','.join(job_cards)} {command}\n")
        log.flush()
        job["proc"] = subprocess.Popen(
            [str(tree / "tools/enroot_test.sh"), "bash", "-c", command],
            env=dict(env0, ROCR_VISIBLE_DEVICES=",".join(job_cards)),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        job["start"], job["log"], job["cards"] = time.time(), log, job_cards
        running.append(job)
        what = "precompile" if pre else "start"
        shared = "" if job["gpus"] else " (compile-only, shared)"
        print(f"{time.strftime('%H:%M:%S')} {what} {t['id']} on GPUs {','.join(job_cards)}{shared}", flush=True)

    busy, running, done = set(), [], []
    compile_card = None  # the card the running compile-only jobs share
    while lane or pending or running:
        sharing = sum(1 for r in running if not r["gpus"])
        while lane and sharing < COMPILE_LANES:
            if not sharing:
                free = [c for c in cards if c not in busy]
                if not free:
                    break
                compile_card = free[0]
                busy.add(compile_card)
            start(lane.pop(0), [compile_card])
            sharing += 1
        for t in list(pending):
            free = [c for c in cards if c not in busy]
            if t["id"] in waiting or t["gpus"] > len(free):
                continue
            pending.remove(t)
            busy.update(free[: t["gpus"]])
            start(t, free[: t["gpus"]])
        time.sleep(2)
        for j in list(running):
            rc = j["proc"].poll()
            limit = PRECOMPILE_TIMEOUT_S if "of" in j else TARGET_TIMEOUT_S
            if rc is None and time.time() - j["start"] > limit:
                os.killpg(j["proc"].pid, 9)
                rc = j["proc"].wait()
                j["log"].write(f"\nkilled after {limit} s\n")
            if rc is None:
                continue
            seconds = round(time.time() - j["start"])
            j["log"].close()
            running.remove(j)
            if j["gpus"] or not any(not r["gpus"] for r in running):
                busy.difference_update(j["cards"])
            if "of" in j:  # its exit code is not a result: tests fail in compile-only mode
                t = j["of"]
                t["precompile_rc"], t["precompile_seconds"] = rc, seconds
                waiting.discard(t["id"])
                print(f"{time.strftime('%H:%M:%S')} precompiled {t['id']} (rc {rc}) in {seconds} s", flush=True)
                continue
            j["rc"], j["seconds"] = rc, seconds
            done.append(j)
            print(f"{time.strftime('%H:%M:%S')} done  {j['id']} rc {rc} in {seconds} s", flush=True)
    keep = ("id", "gpus", "command", "cards", "rc", "seconds", "precompile", "precompile_rc", "precompile_seconds")
    (out / "summary.json").write_text(json.dumps([{k: t.get(k) for k in keep} for t in done], indent=1))
    sys.exit(1 if any(t["rc"] for t in done) else 0)


def junit(path):
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def report(a):
    out = Path(a.out)
    s = out / "summary.json"
    if not s.is_file():
        print(f"No summary in {out}: the Slurm job did not finish (see slurm.log there).")
        sys.exit(1)
    data = json.loads(s.read_text())
    if isinstance(data, dict):
        print(f"The Slurm job stopped before the tests: {data.get('error')}.")
        sys.exit(1)
    bad = [t for t in data if t["rc"]]
    print(f"### Kernel tests: {len(data) - len(bad)} / {len(data)} targets passed\n")
    print("| target | GPUs | result | tests | failed | errors | skipped | minutes |")
    print("|---|---|---|---|---|---|---|---|")
    for t in sorted(data, key=lambda t: (t["rc"] == 0, t["id"])):
        j = junit(out / (slug(t["id"]) + ".xml")) or {}
        res = "pass" if t["rc"] == 0 else f"**FAIL** (rc {t['rc']})"
        cols = [str(j.get(k, "-")) for k in ("tests", "failures", "errors", "skipped")]
        print(f"| `{t['id']}` | {t['gpus']} | {res} | {' | '.join(cols)} | {t['seconds'] / 60:.1f} |")
    for t in bad:
        log = out / (slug(t["id"]) + ".log")
        tail = log.read_text(errors="replace").splitlines()[-40:] if log.is_file() else []
        print(f"\n<details><summary>{t['id']}: last lines of the log</summary>\n\n```\n" + "\n".join(tail) + "\n```\n</details>")
    if a.durations:
        d = Path(a.durations)
        old = json.loads(d.read_text()) if d.is_file() else {}
        old.update({t["id"]: t["seconds"] for t in data if t["rc"] == 0})
        tmp = d.with_suffix(".tmp")
        tmp.write_text(json.dumps(old, indent=1, sort_keys=True))
        tmp.replace(d)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--mode", choices=["changed", "all"], required=True)
    p.add_argument("--base", default="origin/main")
    r = sub.add_parser("run")
    r.add_argument("--targets", required=True)
    r.add_argument("--tree", required=True, help="the unpacked snapshot on the node-local disk")
    r.add_argument("--trees", required=True, help="node-local dir for the per-target copies")
    r.add_argument("--out", required=True)
    r.add_argument("--durations", required=True)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--durations")
    a = ap.parse_args()
    {"plan": plan, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
