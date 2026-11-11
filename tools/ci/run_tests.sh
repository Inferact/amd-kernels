#!/usr/bin/env bash
# The kernel-test pipeline, from the CI runner on a Slurm submit host (or by hand from a checkout there): pick the
# targets, snapshot the commit, run them in one 8-GPU Slurm job (tools/ci/node_job.sh packs them onto the node's GPUs),
# and print the summary. Exit 1 when a target fails or the job does not finish.
#   tools/ci/run_tests.sh changed <base rev>   # a pull request: the entries it touches (+ the repo checks)
#   tools/ci/run_tests.sh all                  # main, nightly, by hand
# Runs land in $AKR_CI_ROOT/runs/<time>-<sha7>-<run id>/: targets.tsv, tree.tar, slurm.log, one .log and .xml per
# target, summary.json, report.md (+ report.rc). A cancelled run (a newer push to the PR) cancels its Slurm job.
# Environment (the job gets it too: node_job.sh, tools/enroot_test.sh):
#   AKR_CI_ROOT        required: the CI directory on storage this host and the GPU nodes share (runs/, wheels/,
#                      durations.json)
#   AKR_CI_PARTITION   optional: the Slurm partition (default: the cluster's default partition)
#   AKR_CI_CPUS, AKR_CI_MEM_PER_GPU, AKR_CI_TIME   optional: the job's CPUs, memory per GPU and time limit
set -euo pipefail
mode=${1:?usage: run_tests.sh changed BASE | all}
base=${2:-}
[[ $mode == all || -n $base ]] || { echo "changed needs a base revision" >&2; exit 2; }
root=$(git rev-parse --show-toplevel)
cd "$root"
sha=$(git rev-parse HEAD)
ci=${AKR_CI_ROOT:?set AKR_CI_ROOT to the CI directory the GPU nodes share}
run=$ci/runs/$(date -u +%Y%m%d-%H%M%S)-${sha:0:7}-${GITHUB_RUN_ID:-local}
mkdir -p "$run"
if [[ $mode == all ]]; then
    python3 tools/ci/ci.py plan --mode all > "$run/targets.tsv"
else
    python3 tools/ci/ci.py plan --mode changed --base "$base" > "$run/targets.tsv"
fi
echo "commit $sha, $(wc -l < "$run/targets.tsv") targets, run dir $run"
cut -f1,2 "$run/targets.tsv" | sed 's/^/  /'
git archive HEAD > "$run/tree.tar"

# nodes with less than 8 x 240G of memory need a smaller AKR_CI_MEM_PER_GPU
partition=()
[[ -z ${AKR_CI_PARTITION:-} ]] || partition=(-p "$AKR_CI_PARTITION")
jid=$(sbatch --parsable "${partition[@]}" --gres=gpu:8 -c "${AKR_CI_CPUS:-96}" \
    --mem-per-gpu="${AKR_CI_MEM_PER_GPU:-240G}" \
    -t "${AKR_CI_TIME:-1:30:00}" -J "akr-ci-${sha:0:7}" -o "$run/slurm.log" --chdir=/tmp \
    --export=ALL,RUN="$run",CI_ROOT="$ci" "$root/tools/ci/node_job.sh")
echo "Slurm job $jid"
echo "$jid" > "$run/slurm_job_id"  # for the workflow's cancel step
# A cancelled workflow run gets SIGINT / SIGTERM and a SIGKILL ~10 s later: cancel the job at once. The waits below
# are `sleep & wait`, which a signal interrupts (a foreground sleep would hold the trap until it ends).
trap 'scancel "$jid" 2>/dev/null || true' EXIT
trap 'echo "cancelled: scancel $jid"; scancel "$jid" 2>/dev/null; exit 130' INT TERM

# The job's own log (the node, then one line per target start / done; each target's output is in its own .log in
# the run dir) is streamed into this log as it grows.
state="" shown=0
follow() {
    local n
    [[ -f "$run/slurm.log" ]] || return 0  # not before the job starts
    n=$(wc -l < "$run/slurm.log")
    if (( n > shown )); then sed -n "$((shown + 1)),${n}p" "$run/slurm.log"; shown=$n; fi
}
while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
    s=$(squeue -h -j "$jid" -o '%T %R' 2>/dev/null || true)
    [[ "$s" == "$state" ]] || { echo "$(date -u +%T) job $jid: $s"; state=$s; }
    follow
    sleep 10 &
    wait $!
done
trap - EXIT
follow
echo "$(date -u +%T) job $jid ended: $(sacct -n -X -j "$jid" -o State,ExitCode 2>/dev/null | xargs || true)"

# node_job.sh writes the report on the node (report.rc = its exit code); a job that died before it gets one here
if [[ -s "$run/report.rc" ]]; then
    rc=$(cat "$run/report.rc")
else
    rc=0
    python3 tools/ci/ci.py report --out "$run" --durations "$ci/durations.json" > "$run/report.md" || rc=$?
fi
cat "$run/report.md"
[[ -z ${GITHUB_STEP_SUMMARY:-} ]] || cat "$run/report.md" >> "$GITHUB_STEP_SUMMARY"
exit $rc
