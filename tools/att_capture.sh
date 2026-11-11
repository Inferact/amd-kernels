#!/usr/bin/env bash
# Capture an ATT (rocprofv3 advanced thread trace) of the kernels that a command launches. Run it after
# `source env.sh`, from the repository root, on an idle GPU (another process on the CU makes the trace invalid):
#   tools/att_capture.sh <kernel regex> <out dir> python3 benchmarks/gemm/mxfp8_common/benchmark_gemm.py --variant fourwave_nt
# It fetches the trace decoder (tools/att_decoder.sh), writes <out>/att.yaml, compiles with the FlyDSL cache off and
# debug info on (code.json then maps each instruction to a source line), dumps the ISA to <out>/dump/<kernel>/ and
# prints the dispatch directories. ATT_ITERATIONS (default "[1, [1-2]]") and ATT_BUFFER (default 0x6000000, 96 MB
# per shader engine; 0xC000000 when the trace is truncated) change the job.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: att_capture.sh KERNEL_REGEX OUT_DIR COMMAND..." >&2; exit 2; }
regex=$1
out=$2
shift 2
lib=$(bash "$(dirname "${BASH_SOURCE[0]}")/att_decoder.sh")
mkdir -p "$out"
out=$(cd "$out" && pwd)
rm -rf "$out/att" "$out/dump"
cat > "$out/att.yaml" <<YAML
jobs:
  - kernel_include_regex: $regex
    kernel_iteration_range: "${ATT_ITERATIONS:-[1, [1-2]]}"
    output_file: att
    output_directory: $out/att
    output_format: [csv]
    truncate_kernels: true
    sys_trace: true
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0xf"
    att_simd_select: "0xf"
    att_buffer_size: "${ATT_BUFFER:-0x6000000}"
    att_library_path: ["$lib"]
YAML
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$out/dump" \
    rocprofv3 -i "$out/att.yaml" -- "$@"
dirs=$(find "$out/att" -maxdepth 3 -type d -name 'ui_output_agent_*_dispatch_*' | sort)
[[ -n "$dirs" ]] || { echo "no dispatch matched '$regex'; list the kernel names with rocprofv3 --stats --kernel-trace" >&2; exit 3; }
echo "dispatch directories:"
echo "$dirs"
echo "ISA:"
find "$out/dump" -name '*_final_isa.s' | sort
