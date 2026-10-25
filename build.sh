#!/usr/bin/env bash
# Builds jdll/oracle_train.jdll, the native library, for this machine.
#
# The Jello headers come from $JELLO_ROOT/include, else from beside the `jello` on PATH: the
# include/ of an installed Jello, or vm/src/include in a checkout of Jello's repository. CC
# chooses the C compiler (cc by default; llvm-mingw's clang on Windows), and CFLAGS adds flags.
#
# ONNX Runtime, with its training API, is found, in order, through pkg-config (libonnxruntime);
# under $ONNXRUNTIME_ROOT, a directory holding include/ and lib/ such as an unpacked
# onnxruntime-training release from https://github.com/microsoft/onnxruntime/releases; or in
# third_party/onnxruntime-training beside this script. Each counts only when it has
# onnxruntime_training_c_api.h, which the plain inference releases for macOS and Linux, and
# Homebrew's onnxruntime, lack. When none has it, the onnxruntime-training release
# $ORT_VERSION for this system and CC's architecture is downloaded with curl, checked against
# the SHA-256 pinned below, and unpacked into third_party/onnxruntime-training; the archive is
# kept in build/. Microsoft publishes no such release for macOS: there,
# scripts/build-onnxruntime.sh builds one from source into third_party/onnxruntime-training.
# The library is linked with its directory as a run path, so it need not be on the loader's
# search path when a program runs; on Windows, put onnxruntime.dll beside the program or on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ORT_VERSION=1.19.2
ORT_RELEASES=https://github.com/microsoft/onnxruntime/releases/download
CC="${CC:-cc}"

die() {
  echo "build.sh: $*" >&2
  exit 1
}

include_dir() {
  local jello candidate
  if [[ -n "${JELLO_ROOT:-}" ]]; then
    echo "$JELLO_ROOT/include"
    return
  fi
  jello="$(command -v jello)" || die "jello is not on PATH; set JELLO_ROOT to Jello's root"
  for candidate in "$(dirname "$jello")/../include" "$(dirname "$jello")/../src/include"; do
    if [[ -f "$candidate/jello/jdll.h" ]]; then
      echo "$candidate"
      return
    fi
  done
  die "cannot find jello/jdll.h beside $jello; set JELLO_ROOT to Jello's root"
}

# Sets ORT_INCLUDE to the first of the directories given that holds ONNX Runtime's headers with
# its training API; fails when none does.
ort_headers() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate/onnxruntime_c_api.h" && -f "$candidate/onnxruntime_training_c_api.h" ]]; then
      ORT_INCLUDE="$candidate"
      return
    fi
  done
  return 1
}

# Sets ORT_INCLUDE and ORT_LIB from a prefix holding ONNX Runtime with its training API; fails
# when it holds none.
ort_from_prefix() {
  ORT_LIB="$1/lib"
  ort_headers "$1/include/onnxruntime" "$1/include" && [[ -d "$ORT_LIB" ]]
}

# Prints the name of the onnxruntime-training release asset for this system and CC's
# architecture, or fails.
ort_asset() {
  local arch
  arch="$("$CC" -dumpmachine 2>/dev/null)" || arch="$(uname -m)"
  case "${arch%%-*}" in
    arm64 | aarch64) arch=arm64 ;;
    x86_64 | amd64) arch=x64 ;;
    i386 | i486 | i586 | i686) arch=x86 ;;
    *) return 1 ;;
  esac
  case "$(uname -s)-$arch" in
    Linux-arm64) echo "onnxruntime-training-linux-aarch64-$ORT_VERSION.tgz" ;;
    Linux-x64) echo "onnxruntime-training-linux-x64-$ORT_VERSION.tgz" ;;
    MINGW*-arm64 | MSYS*-arm64 | CYGWIN*-arm64) echo "onnxruntime-training-win-arm64-$ORT_VERSION.zip" ;;
    MINGW*-x64 | MSYS*-x64 | CYGWIN*-x64) echo "onnxruntime-training-win-x64-$ORT_VERSION.zip" ;;
    MINGW*-x86 | MSYS*-x86 | CYGWIN*-x86) echo "onnxruntime-training-win-x86-$ORT_VERSION.zip" ;;
    *) return 1 ;;
  esac
}

# The SHA-256 of each release asset, taken from the archives on the releases page.
ort_sha256() {
  case "$1" in
    onnxruntime-training-linux-aarch64-1.19.2.tgz) echo a97f0e86c82339f388a65b27d357cd07eefe95fa3e98484102d7ada79d4bf9a2 ;;
    onnxruntime-training-linux-x64-1.19.2.tgz) echo 99e4d88299d80417b31a06e4197c15b56eaaab41c06e891c3368f2512f4952a0 ;;
    onnxruntime-training-win-arm64-1.19.2.zip) echo 1900465e2285c41e97a61a941ec4642da39ab182286712edf8173a9926bc6436 ;;
    onnxruntime-training-win-x64-1.19.2.zip) echo f3d405387fc47256f74f22655ac2f3a0866dc0d989b531fdedb20bdb57a15913 ;;
    onnxruntime-training-win-x86-1.19.2.zip) echo e9c09fba4c4310e1ac871d1d54986b0bcd04c518c70414f7f400a7cece3775fb ;;
    *) return 1 ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Downloads the release asset for this system into build/, checks its SHA-256, and unpacks it
# into the directory given.
download_ort() {
  local dest="$1" asset sha archive actual unpack
  if ! asset="$(ort_asset)"; then
    if [[ "$(uname -s)" == Darwin ]]; then
      die "cannot find ONNX Runtime with its training API, and Microsoft publishes no
onnxruntime-training release for macOS: build it from source into $dest with
$ROOT/scripts/build-onnxruntime.sh, or set ONNXRUNTIME_ROOT to a training build"
    fi
    die "no onnxruntime-training $ORT_VERSION release for $(uname -s) on $("$CC" -dumpmachine 2>/dev/null || uname -m);
build ONNX Runtime with --enable_training_apis and set ONNXRUNTIME_ROOT to it"
  fi
  sha="$(ort_sha256 "$asset")" || die "no SHA-256 pinned for $asset"
  command -v curl >/dev/null 2>&1 || die "curl is needed to download ONNX Runtime"
  mkdir -p "$ROOT/build" "$(dirname "$dest")"
  archive="$ROOT/build/$asset"
  if [[ ! -f "$archive" || "$(sha256_of "$archive")" != "$sha" ]]; then
    echo "build.sh: downloading $asset"
    curl -fsSL --retry 3 -o "$archive.part" "$ORT_RELEASES/v$ORT_VERSION/$asset" ||
      die "cannot download $ORT_RELEASES/v$ORT_VERSION/$asset"
    actual="$(sha256_of "$archive.part")" || die "cannot compute the SHA-256 of $archive.part"
    [[ "$actual" == "$sha" ]] ||
      die "$asset has SHA-256 $actual, not the pinned $sha; it is left in $archive.part"
    mv "$archive.part" "$archive"
  fi
  echo "build.sh: $asset matches its pinned SHA-256; unpacking it into $dest"
  unpack="$(mktemp -d "$ROOT/build/unpack.XXXXXX")" || die "cannot make a directory in build/"
  case "$asset" in
    *.zip)
      command -v unzip >/dev/null 2>&1 || die "unzip is needed to unpack $asset"
      unzip -q "$archive" -d "$unpack" || die "cannot unpack $archive"
      ;;
    *) tar -xzf "$archive" -C "$unpack" || die "cannot unpack $archive" ;;
  esac
  mv "$unpack/${asset%.*}" "$dest" || die "cannot move the unpacked release to $dest"
  rmdir "$unpack"
}

# Sets ORT_INCLUDE and ORT_LIB, downloading ONNX Runtime's training build when it is nowhere to
# be found.
find_ort() {
  local dir bundled="$ROOT/third_party/onnxruntime-training"
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libonnxruntime; then
    dir="$(pkg-config --variable=includedir libonnxruntime)"
    ORT_LIB="$(pkg-config --variable=libdir libonnxruntime)"
    if ort_headers "$dir/onnxruntime" "$dir"; then
      echo "build.sh: ONNX Runtime from pkg-config, in $ORT_LIB"
      return
    fi
  fi
  if [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
    ort_from_prefix "$ONNXRUNTIME_ROOT" ||
      die "ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT holds no include/onnxruntime_training_c_api.h and
lib/: oracle_train needs ONNX Runtime with its training API"
    echo "build.sh: ONNX Runtime from ONNXRUNTIME_ROOT, $ONNXRUNTIME_ROOT"
    return
  fi
  if ort_from_prefix "$bundled"; then
    echo "build.sh: ONNX Runtime from $bundled"
    return
  fi
  [[ ! -e "$bundled" ]] ||
    die "$bundled holds no include/onnxruntime_training_c_api.h and lib/; move it aside to fetch ONNX Runtime again"
  download_ort "$bundled"
  ort_from_prefix "$bundled" || die "the release unpacked into $bundled holds no training API"
}

INCLUDE="$(include_dir)"
find_ort
case "$(uname -s)" in
  Darwin) SHARED=(-bundle "-Wl,-rpath,$ORT_LIB") ;;
  MINGW* | MSYS* | CYGWIN*) SHARED=(-shared) ;;
  *) SHARED=(-fPIC -shared "-Wl,-rpath,$ORT_LIB") ;;
esac

# ONNX Runtime's headers are its own, so they are included as system headers: their warnings
# are not ours to fix.
# shellcheck disable=SC2086
"$CC" -std=c11 -O2 -Wall -Wextra ${CFLAGS:-} "${SHARED[@]}" -I"$INCLUDE" -isystem "$ORT_INCLUDE" \
  "$ROOT/jdll/oracle_train.c" -L"$ORT_LIB" -lonnxruntime -o "$ROOT/jdll/oracle_train.jdll"
echo "built jdll/oracle_train.jdll"
