#!/usr/bin/env bash
# Package the native snakes as their own AWS Lambda function (provided.al2023, arm64): the
# Rust handler (native/crates/lambda, binary slinky-lambda as `bootstrap`) and the az_modes
# weights (weights/az-modes: config.json, params.npz). Output: deploy/lambda-native/dist/native.zip.
#
#   deploy/lambda-native/build.sh
#
# Weights: NATIVE_RUN (default /home/nathan/slinky-serve/az-modes-M-s0). Toolchain: rustup's
# cargo (native/.cargo-env.sh) and cargo-lambda in ~/.local/lambda-tools (see
# native/crates/lambda/README.md). The target dir is CARGO_TARGET_DIR (default
# /dev/shm/naz-target-lambda, in RAM).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
native_run="${NATIVE_RUN:-/home/nathan/slinky-serve/az-modes-M-s0}"
[ -f "$native_run/params.npz" ] && [ -f "$native_run/config.json" ] \
    || { echo "no az_modes run dir at $native_run (set NATIVE_RUN)" >&2; exit 1; }

# shellcheck disable=SC1091
source "$root/native/.cargo-env.sh"  # rustup's cargo (it also sets a target dir: ours wins)
export PATH="$HOME/.local/lambda-tools/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/dev/shm/naz-target-lambda}"
# unwind: a panic in one request answers the safe move instead of killing the process
export CARGO_PROFILE_RELEASE_PANIC=unwind

stage="$(mktemp -d /dev/shm/lambda-native-stage.XXXXXX)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/weights/az-modes"
cp "$native_run/config.json" "$native_run/params.npz" "$stage/weights/az-modes/"
# --include is relative to the current directory and keeps that path inside the zip
(cd "$stage" && cargo lambda build --release --arm64 --manifest-path "$root/native/crates/lambda/Cargo.toml" \
    --bin slinky-lambda --output-format zip --include weights)
mkdir -p "$here/dist"
cp "$CARGO_TARGET_DIR/lambda/slinky-lambda/bootstrap.zip" "$here/dist/native.zip"
python3 -c "import sys, zipfile; n = set(zipfile.ZipFile(sys.argv[1]).namelist()); \
sys.exit(not {'bootstrap', 'weights/az-modes/params.npz', 'weights/az-modes/config.json'} <= n)" "$here/dist/native.zip" \
    || { echo "dist/native.zip is missing bootstrap or the weights" >&2; exit 1; }
ls -l "$here/dist/native.zip"
sha256sum "$here/dist/native.zip"
