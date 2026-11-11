# Source it before running tests and benchmarks (in the current Python environment, no venv): the ROCm SDK, this
# checkout on PYTHONPATH (amd_kernels, tests), and every JIT cache in one directory, <repo>/.cache.
AMD_KERNELS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AMD_KERNELS_REPO
if [[ -z "${ROCM_PATH:-}" ]]; then
  for _sdk in /opt/rocm-7.2.3 /opt/rocm-7.2.2 /opt/rocm; do
    if [[ -x "$_sdk/bin/hipcc" && -f "$_sdk/include/hip/hip_runtime.h" ]]; then
      ROCM_PATH="$_sdk"; break
    fi
  done
  unset _sdk
fi
if [[ ! -x "${ROCM_PATH:-}/bin/hipcc" || ! -f "${ROCM_PATH:-}/include/hip/hip_runtime.h" ]]; then
  echo "env.sh: set ROCM_PATH to a ROCm 7.2 SDK" >&2
  return 1
fi
if [[ ! -r "$ROCM_PATH/.info/version" || "$(<"$ROCM_PATH/.info/version")" != 7.2.* ]]; then
  echo "env.sh: ROCM_PATH must point to a ROCm 7.2 SDK" >&2
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
# One cache directory for every kernel.
# TORCHINDUCTOR_CACHE_DIR also avoids torch's getpass.getuser() default, which fails on nodes that cannot resolve
# the uid.
_cache="$AMD_KERNELS_REPO/.cache"
export XDG_CACHE_HOME="$_cache" AITER_ROOT_DIR="$_cache/aiter" TRITON_CACHE_DIR="$_cache/triton"
export AMD_COMGR_CACHE_DIR="$_cache/comgr" TORCHINDUCTOR_CACHE_DIR="$_cache/inductor"
export FLYDSL_RUNTIME_CACHE_DIR="$_cache/flydsl" MORI_JIT_CACHE_DIR="$_cache/mori"
unset _cache
