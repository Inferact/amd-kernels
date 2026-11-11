#!/usr/bin/env bash
# Run tests or benchmarks inside an enroot container of a base image, never on the host. The image's Python environment
# is the test environment (Python 3.12, torch, AITER, FlyDSL, pytest; no per-kernel venv). Run it on the GPU node
# (inside the Slurm allocation), from a checkout on a node-local disk.
#   tools/enroot_test.sh                                                      # python3 -m pytest tests -q
#   tools/enroot_test.sh python3 -m pytest tests/gemm/mxfp8_common -q
#   tools/enroot_test.sh python3 benchmarks/gemm/mxfp8_common/benchmark_gemm.py --variant decode
# Every command runs from the repository root after `source env.sh` (the checkout on PYTHONPATH, one cache dir).
# Environment:
#   AKR_IMAGE            required: a ROCm PyTorch image with FlyDSL >= 0.5.0 and AITER, as `enroot import docker://`
#                        takes it ([REGISTRY#]REPO[:TAG][@sha256:DIGEST]); a private registry's credentials go in
#                        enroot's credentials file ($ENROOT_CONFIG_PATH/.credentials)
#   ENROOT_TEST_SCRATCH  required: a node-local directory for the image's .sqsh and the rc file
#   AKR_IMAGE_URL        optional: fetch the image's .sqsh (`enroot import` output) over HTTP instead of importing it
#   AKR_IMAGE_SHA256     optional: the sha256 the fetched .sqsh must have
#   AKR_ENROOT_NAME      optional: the container name (default: one per node and image, kept; CI: one per job)
set -euo pipefail
IMAGE="${AKR_IMAGE:?set AKR_IMAGE to a ROCm PyTorch image with FlyDSL >= 0.5.0 and AITER}"
SCRATCH="${ENROOT_TEST_SCRATCH:?set ENROOT_TEST_SCRATCH to a node-local directory}"
IMAGE_URL="${AKR_IMAGE_URL:-}"
if [[ -n "$IMAGE_URL" ]]; then
    SQSH_NAME="${IMAGE_URL##*/}"
else
    SQSH_NAME="${IMAGE##*/}"
    SQSH_NAME="${SQSH_NAME//[^A-Za-z0-9._-]/_}.sqsh"
fi
NAME="${AKR_ENROOT_NAME:-akr-${SQSH_NAME%.sqsh}}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$SCRATCH/images" "$SCRATCH/tools"
# The image, the container rootfs, the checkout and its caches stay on a node-local disk: on the NFS share every
# import and compile-cache read stalls.
for d in "$SCRATCH" "$repo"; do
    if [[ "$(stat -f -c %T "$d")" == nfs* ]]; then
        echo "$d is on NFS; use a node-local disk (ENROOT_TEST_SCRATCH and the checkout)" >&2
        exit 1
    fi
done

sqsh="$SCRATCH/images/$SQSH_NAME"
if [[ ! -f "$sqsh" ]]; then
    if [[ -z "$IMAGE_URL" ]]; then
        enroot import -o "$sqsh.tmp" "docker://$IMAGE"
    else
        echo "fetching $IMAGE_URL" >&2
        curl -fL --retry 3 -o "$sqsh.tmp" "$IMAGE_URL"
        if [[ -n "${AKR_IMAGE_SHA256:-}" ]]; then
            echo "$AKR_IMAGE_SHA256  $sqsh.tmp" | sha256sum -c --quiet - \
                || { rm -f "$sqsh.tmp"; echo "sha256 mismatch for $IMAGE_URL" >&2; exit 1; }
        fi
    fi
    mv "$sqsh.tmp" "$sqsh"
fi
enroot list | grep -qx "$NAME" || enroot create --name "$NAME" "$sqsh"

# --rc skips any image ENTRYPOINT; the rc file must sit on a local disk (enroot copies it with cp -p).
rc="$SCRATCH/tools/norc.sh"
printf '#!/bin/sh\nexec "$@"\n' > "$rc"
chmod +x "$rc"

[[ $# -gt 0 ]] || set -- python3 -m pytest tests -q
envs=()
for v in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    [[ -z "${!v:-}" ]] || envs+=(-e "$v=${!v}")
done
exec enroot start --root --rw --rc "$rc" "${envs[@]}" -m "$repo:$repo" "$NAME" \
    bash -c 'cd "$1" && shift && source env.sh && "$@"' _ "$repo" "$@"
