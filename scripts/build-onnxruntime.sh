#!/usr/bin/env bash
# Builds ONNX Runtime with its training API from source, for macOS, where Microsoft publishes no
# onnxruntime-training release, and installs it into third_party/onnxruntime-training laid out as
# the releases are: include/ with onnxruntime_c_api.h and onnxruntime_training_c_api.h, and lib/
# with libonnxruntime.dylib. build.sh then finds it there.
#
# The source, tag v$ORT_VERSION with its submodules, is cloned shallowly into
# build/onnxruntime-$ORT_VERSION and built in build/onnxruntime-$ORT_VERSION-build, for this
# machine's architecture, as a shared library in its Release configuration. The build needs git,
# Python 3, CMake and Xcode's command line tools, and takes an hour or more. It stops, without
# touching anything, when third_party/onnxruntime-training already exists. It deletes nothing:
# move the source and build directories aside when they are no longer wanted.
#
# --print prints the commands it would run instead of running them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_VERSION=1.19.2
SRC="$ROOT/build/onnxruntime-$ORT_VERSION"
BUILD="$ROOT/build/onnxruntime-$ORT_VERSION-build"
STAGE="$ROOT/build/onnxruntime-training-$ORT_VERSION"
DEST="$ROOT/third_party/onnxruntime-training"
LIB="libonnxruntime.$ORT_VERSION.dylib"

die() {
  echo "build-onnxruntime.sh: $*" >&2
  exit 1
}

PRINT=0
case "${1:-}" in
  "") ;;
  --print) PRINT=1 ;;
  *) die "usage: $0 [--print]" ;;
esac

# Runs a command, or prints it under --print.
run() {
  if [[ "$PRINT" == 1 ]]; then
    printf '%q ' "$@"
    echo
  else
    "$@"
  fi
}

[[ "$(uname -s)" == Darwin ]] ||
  die "this builds ONNX Runtime for macOS; elsewhere, build.sh downloads an onnxruntime-training release"
for dir in "$DEST" "$STAGE"; do
  if [[ -e "$dir" && "$PRINT" == 1 ]]; then
    echo "# $dir already exists, so a real run would stop here"
  elif [[ -e "$dir" ]]; then
    die "$dir already exists; move it aside to build ONNX Runtime again"
  fi
done
ARCH="$(uname -m)"
if [[ "$PRINT" == 0 ]]; then
  for tool in git python3 cmake xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is needed to build ONNX Runtime"
  done
fi

if [[ -d "$SRC/.git" ]]; then
  echo "build-onnxruntime.sh: using the source already in $SRC"
else
  run mkdir -p "$ROOT/build"
  run git clone --depth 1 --branch "v$ORT_VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "$SRC"
fi

run python3 "$SRC/tools/ci_build/build.py" --build_dir "$BUILD" --config Release \
  --build_shared_lib --enable_training_apis --parallel --skip_tests --skip_submodule_sync \
  --compile_no_warning_as_error --osx_arch "$ARCH" \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

# Installs into a staging directory, then lays it out as the releases are, as ONNX Runtime's own
# tools/ci_build/github/linux/copy_strip_binary.sh does, before moving it into place.
run cmake --install "$BUILD/Release" --config Release --prefix "$STAGE"
run sh -c 'mv "$1"/include/onnxruntime/* "$1"/include/ && rmdir "$1"/include/onnxruntime' sh "$STAGE"
run dsymutil "$STAGE/lib/$LIB" -o "$STAGE/lib/$LIB.dSYM"
run strip -S "$STAGE/lib/$LIB"
run cp "$SRC/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h" "$STAGE/include/"
run cp "$SRC/README.md" "$SRC/LICENSE" "$SRC/ThirdPartyNotices.txt" "$SRC/VERSION_NUMBER" "$STAGE/"
run cp "$SRC/docs/Privacy.md" "$STAGE/"
run sh -c 'git -C "$1" rev-parse HEAD > "$2/GIT_COMMIT_ID"' sh "$SRC" "$STAGE"
run mkdir -p "$ROOT/third_party"
run mv "$STAGE" "$DEST"
[[ "$PRINT" == 1 ]] || echo "build-onnxruntime.sh: ONNX Runtime $ORT_VERSION with its training API is in $DEST"
