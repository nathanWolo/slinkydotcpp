"""AlphaZero generation loop: train a new net from the best one, gate it head to head, repeat.

Each generation fine-tunes the current best net with ``baselines/az.py --init-from``
(fresh self-play data from the best net, then learning), then plays ``az-<sims>`` of
the new net against ``az-<sims>`` of the best for ``--games`` seat-balanced duels.
The new net becomes the best if it is significantly better: the lower end of its
score's 95% interval is above ``--gate`` (0.5). Matches are 4096 games by default
(about +-0.015), played in saved chunks by parallel worker processes, so an
interrupted match resumes.
It also plays generation 0 (the starting checkpoint), to track total progress.
The loop stops when ``--patience`` generations in a row fail the gate (a plateau)
or after ``--max-gens``. Re-running it resumes: finished generations are skipped.

    python baselines/az_loop.py                    # CPU defaults
    python baselines/az_loop.py --steps 2097152 -- --num-envs 512   # extra az.py flags after --
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import jax

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from slinky.agents import make_agent, make_env  # noqa: E402
from slinky.evaluate import _summarize, run_match  # noqa: E402
from slinky.types import GameConfig  # noqa: E402

GEN0 = "baselines/checkpoints/az-duel-seed0"


# Games per saved piece of a match, so an interrupted match resumes.
CHUNK = int(os.environ.get("AZ_LOOP_CHUNK", 256))
WORKERS = 3  # processes per match: one JAX process keeps only ~5 cores busy


def match_dir(a: str, b: str, sims: int, seed: int) -> str:
    return os.path.join(REPO, a, f"match-{os.path.basename(b)}-s{sims}-seed{seed}")


def chunks(games: int) -> list[tuple[int, int]]:
    return [(first, min(CHUNK, games - first)) for first in range(0, games, CHUNK)]


def match(a: str, b: str, games: int, sims: int, seed: int, workers: int = WORKERS) -> dict:
    """``az-<sims>`` of run ``a`` vs run ``b``, from a's side.

    The games are split into chunks of game ids, played by ``workers`` processes
    in parallel; each finished chunk is saved under run ``a`` and reused on a
    re-run. Game ``g`` is the same game whatever the chunking (``run_match``).
    """
    d = match_dir(a, b, sims, seed)
    os.makedirs(d, exist_ok=True)
    todo = [c for c in chunks(games) if not os.path.exists(os.path.join(d, f"{c[0]}-{c[1]}.json"))]
    if todo:
        args = [a, b, str(sims), str(seed), str(games)]
        procs = [
            subprocess.Popen(
                [sys.executable, "-u", __file__, "--worker", *args, str(k), str(workers)]
            )
            for k in range(min(workers, len(todo)))
        ]
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError(f"a match worker failed: {a} vs {b}")
    parts = {}
    for first, n in chunks(games):
        with open(os.path.join(d, f"{first}-{n}.json")) as f:
            parts[f"{first}:{n}"] = json.load(f)
    out = score_of(parts)
    out["seconds"] = sum(p["seconds"] for p in parts.values())
    return out


def worker(a: str, b: str, sims: int, seed: int, games: int, k: int, workers: int) -> None:
    """Play every ``workers``-th unfinished chunk of a match, starting at chunk ``k``."""
    game = GameConfig()
    pa = make_agent(f"az-{sims}:{a}", game)
    pb = make_agent(f"az-{sims}:{b}", game)
    env = make_env(game, True)
    d = match_dir(a, b, sims, seed)
    for first, n in chunks(games)[k::workers]:
        path = os.path.join(d, f"{first}-{n}.json")
        if os.path.exists(path):
            continue
        t = time.perf_counter()
        r = run_match(
            env, pa.policy, pb.policy, jax.random.key(seed), n, max_turns=1000,
            batch_size=CHUNK, first_game=first,
        ).result  # fmt: skip
        with open(path + ".tmp", "w") as f:
            json.dump({**r._asdict(), "seconds": time.perf_counter() - t}, f)
        os.replace(path + ".tmp", path)
        done = len([x for x in os.listdir(d) if x.endswith(".json")])
        print(f"  {a} vs {os.path.basename(b)}: chunk {first}-{first + n} score {r.score:.3f} "
              f"({done}/{len(chunks(games))} chunks done)", flush=True)  # fmt: skip


def score_of(cache: dict) -> dict:
    """The combined result of saved chunks."""
    w, d, l_, tr = (
        sum(c[k] for c in cache.values()) for k in ("wins", "draws", "losses", "truncated")
    )
    turns = sum(c["mean_turns"] * c["num_games"] for c in cache.values())
    return _summarize(wins=w, draws=d, losses=l_, truncated=tr, total_turns=round(turns))._asdict()


def train(gen_dir: str, init_from: str, steps: int, seed: int, lr: float, extra: list[str]) -> None:
    cmd = [
        sys.executable, "baselines/az.py",
        "--run-dir", gen_dir, "--init-from", init_from, "--seed", str(seed),
        "--total-env-steps", str(steps), "--buffer-capacity", "262144",
        "--learning-starts", "32768", "--lr", str(lr), "--final-lr-frac", "0.1",
        "--eval-games", "0", "--save-replay", "false", "--checkpoint-every", str(steps),
        *extra,
    ]  # fmt: skip
    with open(os.path.join(REPO, gen_dir + ".log"), "a") as log:
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    if sys.argv[1:2] == ["--worker"]:
        a, b, sims, seed, games, k, n = sys.argv[2:9]
        worker(a, b, int(sims), int(seed), int(games), int(k), int(n))
        return
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/loop")
    p.add_argument("--steps", type=int, default=1048576, help="env steps per generation")
    p.add_argument("--games", type=int, default=4096, help="games per gating match")
    p.add_argument("--sims", type=int, default=64, help="search simulations in matches")
    p.add_argument("--lr", type=float, default=1e-4, help="peak learning rate (decays to 0.1x)")
    p.add_argument(
        "--gate", type=float, default=0.5,
        help="accept when the score's 95%% lower bound is above this",
    )  # fmt: skip
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-gens", type=int, default=12)
    p.add_argument("extra", nargs="*", help="extra az.py flags (after --)")
    args = p.parse_args()

    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    history_path = os.path.join(REPO, args.out, "history.jsonl")
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = [json.loads(line) for line in f if line.strip()]
    best, fails = GEN0, 0
    for h in history:
        best, fails = (h["gen_dir"], 0) if h["accepted"] else (best, fails + 1)

    for g in range(len(history) + 1, args.max_gens + 1):
        if fails >= args.patience:
            break
        gen_dir = f"{args.out}/gen-{g:02d}"
        print(f"gen {g}: training from {best} ({args.steps:,} env steps)", flush=True)
        t = time.perf_counter()
        config_path = os.path.join(REPO, gen_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                trained_from = json.load(f).get("init_from")
            if trained_from != best:  # a leftover trained from another net: set it aside
                aside = f"{gen_dir}-from-{os.path.basename(trained_from or 'scratch')}"
                os.replace(os.path.join(REPO, gen_dir), os.path.join(REPO, aside))
                os.replace(os.path.join(REPO, gen_dir + ".log"), os.path.join(REPO, aside + ".log"))
                print(f"gen {g}: {gen_dir} was trained from {trained_from}; moved to {aside}")
        if not os.path.exists(os.path.join(REPO, gen_dir, "params.npz")):
            train(gen_dir, best, args.steps, seed=g, lr=args.lr, extra=args.extra)
        train_s = time.perf_counter() - t

        vs_best = match(gen_dir, best, args.games, args.sims, seed=1000 + g)
        vs_gen0 = vs_best if best == GEN0 else match(gen_dir, GEN0, args.games, args.sims, 2000 + g)
        accepted = vs_best["score"] - vs_best["score_ci95"] > args.gate
        record = {
            "gen": g, "gen_dir": gen_dir, "init_from": best, "env_steps": args.steps,
            "train_seconds": train_s, "vs_best": vs_best, "vs_gen0": vs_gen0,
            "accepted": accepted, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }  # fmt: skip
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"gen {g}: vs best {vs_best['score']:.3f} ±{vs_best['score_ci95']:.3f}, "
            f"vs gen0 {vs_gen0['score']:.3f} ±{vs_gen0['score_ci95']:.3f} "
            f"-> {'ACCEPTED' if accepted else 'rejected'} (train {train_s / 60:.0f} min)",
            flush=True,
        )
        best, fails = (gen_dir, 0) if accepted else (best, fails + 1)

    reason = "plateau" if fails >= args.patience else "max generations"
    print(f"stopped ({reason}); best: {best}", flush=True)
    with open(os.path.join(REPO, args.out, "BEST"), "w") as f:
        f.write(best + "\n")


if __name__ == "__main__":
    main()
