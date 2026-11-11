#!/bin/bash
# The Slurm side of tools/ci/run_tests.sh (one 8-GPU job): unpack the snapshot $RUN/tree.tar on the node-local disk
# and pack $RUN/targets.tsv onto the job's GPUs (tools/ci/ci.py run). Nothing an earlier run left is reused: the
# per-target trees (trees-<job>/) start with empty JIT caches, and the job runs in its own container, created from
# the image (enroot create, ~10 s for the 28 GB rootfs on the node-local disk) and removed at the end. A container
# that tools/enroot_test.sh keeps across runs would carry what earlier runs wrote into it (AITER's JIT modules in
# site-packages, pip installs). AKR_CI_SCRATCH (required) is a node-local directory; the trees go in its akr-ci/.
set -euo pipefail
: "${RUN:?}" "${CI_ROOT:?}"
d=${AKR_CI_SCRATCH:?set AKR_CI_SCRATCH to a node-local directory}/akr-ci
base=$d/base-$SLURM_JOB_ID
trees=$d/trees-$SLURM_JOB_ID
export AKR_ENROOT_NAME=akr-ci-$SLURM_JOB_ID
# leftovers of cancelled jobs: a CI job holds the whole node, so no other akr-ci-* container of ours is in use
for c in $(enroot list | grep '^akr-ci-' || true); do enroot remove -f "$c" > /dev/null; done
rm -rf "$d"/base-* "$d"/trees-*
mkdir -p "$base" "$trees"
trap 'enroot remove -f "$AKR_ENROOT_NAME" > /dev/null 2>&1; rm -rf "$base" "$trees"' EXIT
tar x -C "$base" -f "$RUN/tree.tar"
# AMD cards: Slurm may set only CUDA_VISIBLE_DEVICES
if [[ -z "${ROCR_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export ROCR_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
echo "node $(hostname) job $SLURM_JOB_ID ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
# pytest-xdist for the precompile commands with -n (targets.tsv), from the wheels in $CI_ROOT/wheels/ (no network).
# Without it such a precompile fails at once and its target runs cold.
if compgen -G "$CI_ROOT/wheels/*.whl" > /dev/null; then
    t=$(date +%s)
    mkdir -p "$base/.ci/wheels"
    cp "$CI_ROOT"/wheels/*.whl "$base/.ci/wheels/"
    if "$base/tools/enroot_test.sh" python3 -m pip install -q --root-user-action=ignore --no-index \
        --find-links .ci/wheels pytest-xdist; then
        echo "pytest-xdist installed ($(( $(date +%s) - t )) s)"
    else
        echo "pytest-xdist not installed"
    fi
else
    echo "no wheels in $CI_ROOT/wheels/: pytest-xdist not installed"
fi
rc=0
python3 "$base/tools/ci/ci.py" run --targets "$RUN/targets.tsv" --tree "$base" --trees "$trees" --out "$RUN" \
    --durations "$CI_ROOT/durations.json" || rc=$?
# The report is written here, not on the submit host: there the first open of each file this node just wrote on NFS
# can take ~1.2 s (54 junit files: 63 s); run_tests.sh reads report.md and report.rc.
report_rc=0
python3 "$base/tools/ci/ci.py" report --out "$RUN" --durations "$CI_ROOT/durations.json" > "$RUN/report.md" \
    || report_rc=$?
echo "$report_rc" > "$RUN/report.rc"
exit $rc
