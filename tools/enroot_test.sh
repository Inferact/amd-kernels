#!/usr/bin/env bash
# Run tests or benchmarks inside an enroot container of the vLLM ROCm nightly image, never on the host. The image's
# Python environment is the test environment (its Python, torch, AITER, FlyDSL; no per-kernel venv). Run it on the GPU
# node (inside the Slurm allocation), from a checkout on a node-local disk.
#   tools/enroot_test.sh                                                      # python3 -m pytest tests -q
#   tools/enroot_test.sh python3 -m pytest tests/gemm/mxfp8_common -q
#   tools/enroot_test.sh python3 benchmarks/gemm/mxfp8_common/benchmark_gemm.py --variant decode
#   tools/enroot_test.sh --resolve        # print the image pinned to the digest its tag points to now, and exit
# Every command runs from the repository root after `source env.sh` (the checkout on PYTHONPATH, one cache dir).
# Environment:
#   AKR_IMAGE            optional: the image (default vllm/vllm-openai-rocm:nightly, retagged every day), as `enroot
#                        import docker://` takes it ([REGISTRY#]REPO[:TAG][@sha256:DIGEST]). A Docker Hub tag is
#                        resolved to the digest it points to now, and the .sqsh is kept per digest, so a stale local
#                        copy is never used; without Docker Hub the newest local copy of the tag runs, with a warning.
#                        A private registry's credentials go in enroot's credentials file
#                        ($ENROOT_CONFIG_PATH/.credentials)
#   ENROOT_TEST_SCRATCH  required: a node-local directory for the image's .sqsh and the rc file
#   AKR_IMAGE_URL        optional: fetch the image's .sqsh (`enroot import` output) over HTTP instead of importing it
#   AKR_IMAGE_SHA256     optional: the sha256 the fetched .sqsh must have
#   AKR_ENROOT_NAME      optional: the container name (default: one per node and image, kept; CI: one per job)
set -euo pipefail
IMAGE="${AKR_IMAGE:-vllm/vllm-openai-rocm:nightly}"
IMAGE_URL="${AKR_IMAGE_URL:-}"

# The digest Docker Hub serves for REPO:TAG now (a HEAD request: it does not count as a pull).
hub_digest() {
    local repo=$1 tag=$2 token
    [[ $repo == */* ]] || repo=library/$repo
    token=$(curl -fsS --max-time 10 \
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$repo:pull" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])') || return 1
    curl -fsSI --max-time 10 -H "Authorization: Bearer $token" \
        -H "Accept: application/vnd.oci.image.index.v1+json" \
        -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
        -H "Accept: application/vnd.oci.image.manifest.v1+json" \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
        "https://registry-1.docker.io/v2/$repo/manifests/$tag" \
        | tr -d '\r' | awk 'tolower($1) == "docker-content-digest:" {print $2}' | grep -m1 '^sha256:'
}

# A Docker Hub REPO:TAG (no registry part, no digest) becomes REPO:TAG@DIGEST.
ref=${IMAGE%@*}
digest=""
[[ $IMAGE == *@* ]] && digest=${IMAGE##*@}
image_repo=${ref%:*}
[[ $ref == *:* ]] && tag=${ref##*:} || tag=latest
if [[ -z $IMAGE_URL && -z $digest && $image_repo != *#* && ${image_repo%%/*} != *[.:]* ]]; then
    if digest=$(hub_digest "$image_repo" "$tag"); then
        IMAGE="$image_repo:$tag@$digest"
    else
        digest=""
        echo "enroot_test.sh: cannot reach Docker Hub for $image_repo:$tag; using the newest local copy" >&2
    fi
fi
if [[ ${1:-} == --resolve ]]; then
    echo "$IMAGE"
    exit 0
fi
SCRATCH="${ENROOT_TEST_SCRATCH:?set ENROOT_TEST_SCRATCH to a node-local directory}"
stem="${image_repo##*[/#]}_$tag"
stem="${stem//[^A-Za-z0-9._-]/_}"
if [[ -n "$IMAGE_URL" ]]; then
    SQSH_NAME="${IMAGE_URL##*/}"
elif [[ -n "$digest" ]]; then
    SQSH_NAME="$stem-${digest:7:12}.sqsh"
else
    SQSH_NAME=$(cd "$SCRATCH/images" 2>/dev/null && ls -t "$stem"-*.sqsh 2>/dev/null | head -1 || true)
    SQSH_NAME=${SQSH_NAME:-$stem.sqsh}
fi
NAME="${AKR_ENROOT_NAME:-akr-${SQSH_NAME%.sqsh}}"
echo "image $IMAGE" >&2

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
# Calls on one node run at once (the CI's targets, two jobs on one node): one imports the image and one creates the
# container, the others wait for it.
exec 9> "$sqsh.lock"
flock 9
if [[ ! -f "$sqsh" ]]; then
    rm -f "$sqsh.tmp"  # an import a killed call left (we hold the lock)
    if [[ -z "$IMAGE_URL" ]]; then
        src=$IMAGE
        [[ -z $digest ]] || src=$image_repo@$digest
        enroot import -o "$sqsh.tmp" "docker://$src"
    else
        echo "fetching $IMAGE_URL" >&2
        curl -fL --retry 3 -o "$sqsh.tmp" "$IMAGE_URL"
        if [[ -n "${AKR_IMAGE_SHA256:-}" ]]; then
            echo "$AKR_IMAGE_SHA256  $sqsh.tmp" | sha256sum -c --quiet - \
                || { rm -f "$sqsh.tmp"; echo "sha256 mismatch for $IMAGE_URL" >&2; exit 1; }
        fi
    fi
    mv "$sqsh.tmp" "$sqsh"
    # keep this digest's .sqsh and the one before it (a container needs its .sqsh only at enroot create)
    (cd "$SCRATCH/images" && ls -t "$stem"-*.sqsh 2>/dev/null | tail -n +3 | xargs -r rm -f)
fi
enroot list | grep -qx "$NAME" || enroot create --name "$NAME" "$sqsh"
exec 9>&-

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
