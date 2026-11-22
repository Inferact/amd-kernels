# Source it before running tests and benchmarks (in the current Python environment, no venv): the ROCm SDK, this
# checkout on PYTHONPATH (amd_kernels, tests), and every JIT cache in one directory, <repo>/.cache.
AMD_KERNELS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AMD_KERNELS_REPO
if [[ -z "${ROCM_PATH:-}" ]]; then
  for _sdk in /opt/rocm; do
    if [[ -x "$_sdk/bin/hipcc" && -f "$_sdk/include/hip/hip_runtime.h" ]]; then
      ROCM_PATH="$_sdk"; break
    fi
  done
  unset _sdk
fi
if [[ ! -x "${ROCM_PATH:-}/bin/hipcc" || ! -f "${ROCM_PATH:-}/include/hip/hip_runtime.h" ]]; then
  echo "env.sh: set ROCM_PATH to the image's ROCm SDK (bin/hipcc, include/hip)" >&2
  return 1
fi
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$PATH"
export PYTHONPATH="$AMD_KERNELS_REPO${PYTHONPATH:+:$PYTHONPATH}"
# FlyDSL 0.5.0 or newer (it keys a compiled launcher by its traced MLIR). An image with an older FlyDSL gets
# flydsl==$AKR_FLYDSL from PyPI into <repo>/.cache/flydsl-<version>, ahead of the image's on PYTHONPATH.
AKR_FLYDSL="${AKR_FLYDSL:-0.5.0}"
if ! python3 -c 'import re, sys, flydsl
v = tuple(int(x) for x in re.findall(r"\d+", flydsl.__version__.split("+")[0])[:3])
sys.exit(v < (0, 5, 0))' 2>/dev/null; then
  _fly="$AMD_KERNELS_REPO/.cache/flydsl-$AKR_FLYDSL"
  if [[ ! -f "$_fly/.installed" ]]; then
    rm -rf "$_fly.tmp.$$"
    # the image pins its own FlyDSL (pip config / constraints): install from PyPI without them
    PIP_CONFIG_FILE=/dev/null PIP_CONSTRAINT= PIP_ROOT_USER_ACTION=ignore python3 -m pip install -q --no-deps --target "$_fly.tmp.$$" \
      "flydsl==$AKR_FLYDSL" \
      && touch "$_fly.tmp.$$/.installed" && rm -rf "$_fly" && mv "$_fly.tmp.$$" "$_fly" \
      || { echo "env.sh: cannot install flydsl==$AKR_FLYDSL into $_fly" >&2; return 1; }
  fi
  export PYTHONPATH="$_fly:$PYTHONPATH"
  unset _fly
fi
# The test image is Docker Hub's vllm/vllm-openai-rocm:nightly, retagged every day: warn when this environment's vLLM
# is not the newest nightly's commit (Docker Hub asked at most once an hour; no network, no vLLM: no check).
python3 - "$AMD_KERNELS_REPO/.cache/nightly-tag" <<'PY' || true
import json, os, re, sys, time, urllib.request
from importlib import metadata
try:
    local = re.search(r"\+g([0-9a-f]{7,40})", metadata.version("vllm")).group(1)
except Exception:
    sys.exit(0)
path = sys.argv[1]
try:
    if time.time() - os.path.getmtime(path) > 3600:
        raise OSError
    newest, when = open(path).read().split()
except (OSError, ValueError):
    url = "https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags?page_size=25&ordering=last_updated"
    try:
        tags = json.load(urllib.request.urlopen(url, timeout=5))["results"]
    except Exception:
        sys.exit(0)
    hit = next((t for t in tags if re.fullmatch(r"nightly-[0-9a-f]{40}", t["name"])), None)
    if hit is None:
        sys.exit(0)
    newest, when = hit["name"][len("nightly-"):], hit["last_updated"][:16]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{newest} {when}\n")
if not newest.startswith(local):
    print(f"env.sh: this image's vLLM is {local}, Docker Hub's vllm/vllm-openai-rocm:nightly is {newest[:12]} "
          f"({when} UTC): docker pull vllm/vllm-openai-rocm:nightly (tools/enroot_test.sh imports it)", file=sys.stderr)
PY
# One cache directory for every kernel.
# TORCHINDUCTOR_CACHE_DIR also avoids torch's getpass.getuser() default, which fails on nodes that cannot resolve
# the uid.
_cache="$AMD_KERNELS_REPO/.cache"
export XDG_CACHE_HOME="$_cache" AITER_ROOT_DIR="$_cache/aiter" TRITON_CACHE_DIR="$_cache/triton"
export AMD_COMGR_CACHE_DIR="$_cache/comgr" TORCHINDUCTOR_CACHE_DIR="$_cache/inductor"
export FLYDSL_RUNTIME_CACHE_DIR="$_cache/flydsl" MORI_JIT_CACHE_DIR="$_cache/mori"
unset _cache
