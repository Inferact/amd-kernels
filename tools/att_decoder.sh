#!/usr/bin/env bash
# Print the directory of the ATT trace decoder (librocprof-trace-decoder.so); fetch it into <repo>/.cache first use.
# rocprofv3 1.3.2 captures ATT but ships no decoder ("rocprof-trace-decoder library path
# not found"): pass this directory as `att_library_path` in the rocprofv3 input.yaml. The decoder is AMD's binary
# release (ROCm/rocprof-trace-decoder 0.1.6, manylinux 2.28), pinned by sha256 and not committed.
#   lib=$(tools/att_decoder.sh)
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER=0.1.6
SHA256=57a49c74db7dcd29433888ef5d408ca8e22223f2680c566855e4c3c68da7b36d
NAME="rocprof-trace-decoder-manylinux-2.28-$VER-Linux"
dir="$repo/.cache/rocprof-trace-decoder/$VER"
lib="$dir/$NAME/opt/rocm/lib"
if [[ ! -f "$lib/librocprof-trace-decoder.so" ]]; then
    mkdir -p "$dir"
    curl -fsSL --retry 3 -o "$dir/$NAME.tar.gz" \
        "https://github.com/ROCm/rocprof-trace-decoder/releases/download/$VER/$NAME.tar.gz"
    echo "$SHA256  $dir/$NAME.tar.gz" | sha256sum -c --quiet - \
        || { rm -f "$dir/$NAME.tar.gz"; echo "sha256 mismatch for the decoder release" >&2; exit 1; }
    tar xzf "$dir/$NAME.tar.gz" -C "$dir"
fi
echo "$lib"
