"""Build the C environment: ``.venv-puffer/bin/python puffer/setup.py build_ext --inplace``.

Produces ``puffer/binding.cpython-*.so`` (the module must be named ``binding``: PufferLib's
``env_binding.h`` defines ``PyInit_binding``). ``BATTLESNAKE_DEBUG=1`` builds with
``-O0 -g`` and AddressSanitizer + UndefinedBehaviorSanitizer instead of ``-O3``. Python
must preload the ASan runtime, and libstdc++ too (or ASan aborts when jaxlib loads)::

    LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)" \\
    ASAN_OPTIONS=detect_leaks=0 .venv-puffer/bin/python -m pytest tests/test_puffer_env.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent


def pufferlib_ocean_dir() -> str:
    """The directory holding PufferLib's ``env_binding.h``, without importing pufferlib.

    Importing pufferlib has side effects (it symlinks ``resources`` into the current
    directory), so the package is located instead.
    """
    spec = importlib.util.find_spec("pufferlib")
    if spec is None or spec.origin is None:
        raise SystemExit("pufferlib is not installed in this Python")
    ocean = Path(spec.origin).parent / "ocean"
    if not (ocean / "env_binding.h").exists():
        raise SystemExit(f"env_binding.h not found in {ocean}")
    return str(ocean)


debug = os.environ.get("BATTLESNAKE_DEBUG", "") not in ("", "0")
compile_args = ["-std=c11", "-Wall", "-Wextra"]
link_args: list[str] = []
if debug:
    sanitize = "-fsanitize=address,undefined"
    # Python's own flags make signed overflow wrap (-fno-strict-overflow), which would hide
    # it from UBSan; -fstrict-overflow undoes that for this build.
    compile_args += ["-O0", "-g", "-fno-omit-frame-pointer", "-fstrict-overflow", sanitize]
    link_args += [sanitize]
else:
    compile_args += ["-O3"]

# setuptools writes the in-place module to the current directory. Building from puffer/
# also keeps it away from the repository's pyproject.toml (slinky's src layout).
os.chdir(HERE)
setup(
    name="battlesnake-puffer",
    ext_modules=[
        Extension(
            "binding",
            sources=["binding.c"],
            depends=["battlesnake.h"],
            include_dirs=[pufferlib_ocean_dir(), np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ],
)
