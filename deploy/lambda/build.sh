#!/usr/bin/env bash
# Package the snakes as an AWS Lambda zip (arm64, Python 3.13): the handler,
# snake_numpy.py, each numpy snake's checkpoint (package_checkpoints.py),
# numpy's Linux aarch64 wheel, and for the native snakes (lambda_function.NATIVE)
# the Rust engine bin/slinky-engine (aarch64, panic = unwind) and its weights
# (weights/az-modes: config.json, params.npz). Output: deploy/lambda/dist/snake.zip.
#
#   deploy/lambda/build.sh [python]      # python with pip (default: .venv/bin/python)
#
# Native engine: built with cargo-lambda (zig cross-linker; see
# native/crates/lambda/README.md), or ENGINE=<a prebuilt aarch64 slinky-engine>.
# Weights: NATIVE_RUN (default /home/nathan/slinky-serve/az-modes-M-s0), the
# az_modes run dir of az-256. The build target dir is CARGO_TARGET_DIR (default
# /dev/shm/naz-target-lambda-engine, in RAM).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
python="${1:-$root/.venv/bin/python}"
numpy_version="2.2.6"  # the version the port is tested with
native_run="${NATIVE_RUN:-/home/nathan/slinky-serve/az-modes-M-s0}"
engine="${ENGINE:-}"
[ -f "$native_run/params.npz" ] && [ -f "$native_run/config.json" ] \
    || { echo "no az_modes run dir at $native_run (set NATIVE_RUN)" >&2; exit 1; }
if [ -z "$engine" ]; then
    target="${CARGO_TARGET_DIR:-/dev/shm/naz-target-lambda-engine}"
    # shellcheck disable=SC1091
    source "$root/native/.cargo-env.sh"  # rustup's cargo (it also sets a target dir: ours wins)
    export PATH="$HOME/.local/lambda-tools/bin:$PATH" CARGO_TARGET_DIR="$target"
    # unwind: a panic in one request answers the safe move instead of killing the engine
    CARGO_PROFILE_RELEASE_PANIC=unwind cargo lambda build --release --arm64 \
        --manifest-path "$root/native/crates/lambda/Cargo.toml" --bin slinky-engine
    engine="$CARGO_TARGET_DIR/lambda/slinky-engine/bootstrap"
fi
file -b "$engine" | grep -q "ARM aarch64" || { echo "$engine is not an aarch64 binary" >&2; exit 1; }

build="$here/build"
rm -rf "$build" "$here/dist"
mkdir -p "$build" "$here/dist"

"$python" -m pip install --quiet --target "$build" --no-deps --only-binary=:all: \
    --platform manylinux2014_aarch64 --implementation cp --python-version 3.13 \
    "numpy==$numpy_version"
# numpy's own tests and type stubs are not needed at run time.
rm -rf "$build"/numpy/*/tests "$build"/numpy/tests "$build"/numpy/_pyinstaller "$build"/bin
find "$build" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$build" -name "*.pyi" -delete

cp "$here/lambda_function.py" "$here/snake_numpy.py" "$build/"
"$python" "$here/package_checkpoints.py" "$root/baselines/checkpoints" "$build/checkpoints"
mkdir -p "$build/bin" "$build/weights/az-modes"
install -m 755 "$engine" "$build/bin/slinky-engine"
cp "$native_run/config.json" "$native_run/params.npz" "$build/weights/az-modes/"
echo "  az-256: $engine ($(stat -c %s "$engine") bytes), weights from $native_run"

(cd "$build" && "$python" - <<'EOF'
import os, zipfile
with zipfile.ZipFile("../dist/snake.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, _, files in sorted(os.walk(".")):
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, "."))
EOF
)
ls -l "$here/dist/snake.zip"
sha256sum "$here/dist/snake.zip"
