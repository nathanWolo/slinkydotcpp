"""Copy the checkpoints ``lambda_function.SNAKES`` names into a Lambda build directory.

Each gets its ``config.json`` and its ``params.npz`` without the noisy layers'
scales (``*_sigma``, which only exploration uses), so Rainbow's checkpoints
shrink by half::

    python deploy/lambda/package_checkpoints.py baselines/checkpoints build/checkpoints
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambda_function import SNAKES  # noqa: E402  (loads no network)


def main(src: Path, dst: Path) -> None:
    for name, (folder, _) in SNAKES.items():
        out = dst / folder
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / folder / "config.json", out / "config.json")
        with np.load(src / folder / "params.npz") as z:
            np.savez(out / "params.npz", **{k: z[k] for k in z.files if not k.endswith("_sigma")})
        print(f"  {name}: {folder} ({(out / 'params.npz').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
