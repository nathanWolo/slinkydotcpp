#!/usr/bin/env bash
# Package the numpy snakes as an AWS Lambda zip (arm64, Python 3.13): the
# handler, snake_numpy.py, each snake's checkpoint (package_checkpoints.py)
# and numpy's Linux aarch64 wheel. Output: deploy/lambda/dist/snake.zip.
#
#   deploy/lambda/build.sh [python]      # python with pip (default: .venv/bin/python)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
python="${1:-$root/.venv/bin/python}"
numpy_version="2.2.6"  # the version the port is tested with

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
