"""Adversarial review data for the Rust env crate: env_golden.py's two generators (exact
transitions and whole-game distributions) on configs the main golden set does not cover.

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/review_env.py [--only trans,dist] \\
        [--configs NAME,...] [--scale F] [--out DIR]

Writes ``<out>/env_review/trans/<name>/`` and ``<out>/env_review/dist/<name>.json`` in the formats
documented in env_golden.py. ``crates/env/tests/review.rs`` reads them (it skips when missing).

The configs aim at what the main set leaves out: fixed placement and fixed starting food on
even and odd square boards other than 7 and 11 (8x8, 9x9, 10x10), random placement on small
squares and thin or wide rectangles (6x6, 5x5, 3x3 with every cell taken, 1x11, 11x7), boards
nearly full of food (high minimum_food), spawn chances at the edges (0, 1, 2, 100, 101,
minimum_food 0), constrictor duels on both maps, the royale ruleset on the standard and empty
maps, royale on a 7x7 board with a fast shrink, and max_turns truncation in royale and at 1.

Default out /dev/shm/naz-golden (RAM; never committed).
"""
import argparse, dataclasses, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_golden as eg  # noqa: E402

from slinky import modes  # noqa: E402
from slinky.types import GameConfig, Ruleset  # noqa: E402

T, G = 3_072, 6_144


def configs():
    m = modes.mode_config
    R, C = Ruleset.ROYALE, Ruleset.CONSTRICTOR
    return [
        # name, config, transitions, games for the distributions (0: none)
        ("sq8_4", GameConfig(width=8, height=8, num_snakes=4), T, G),
        ("sq9_2", GameConfig(width=9, height=9, num_snakes=2), T, G),
        ("sq10_3", GameConfig(width=10, height=10, num_snakes=3), T, G),
        ("sq6_4", GameConfig(width=6, height=6, num_snakes=4), T, G),
        ("sq5_2", GameConfig(width=5, height=5, num_snakes=2), T, G),
        ("sq3_4", GameConfig(width=3, height=3, num_snakes=4), T, G),
        ("line_1x11", GameConfig(width=1, height=11, num_snakes=2), T, G),
        ("rect_11x7_4", GameConfig(width=11, height=7, num_snakes=4), T, G),
        ("crowded_5x5", GameConfig(width=5, height=5, num_snakes=2, minimum_food=30,
                                   food_spawn_chance=0), T, G),
        ("crowded_7x7_4", GameConfig(width=7, height=7, num_snakes=4, minimum_food=25,
                                     food_spawn_chance=50), T, G),
        ("chance0_min0", dataclasses.replace(m("duel"), food_spawn_chance=0, minimum_food=0), T, G),
        ("chance1", dataclasses.replace(m("duel"), food_spawn_chance=1, minimum_food=0), T, G),
        ("chance2", dataclasses.replace(m("duel"), food_spawn_chance=2, minimum_food=0), T, G),
        ("chance100", dataclasses.replace(m("duel"), food_spawn_chance=100, minimum_food=0), T, G),
        ("chance101_min2", dataclasses.replace(m("standard"), food_spawn_chance=101,
                                               minimum_food=2), T, G),
        ("constrictor2", m("constrictor", 2), T, G),
        ("constrictor_empty2", GameConfig(num_snakes=2, ruleset=C, map="empty"), T, G),
        ("constrictor_7x7", GameConfig(width=7, height=7, num_snakes=4, ruleset=C), T, G),
        ("royale_stdmap", GameConfig(num_snakes=4, ruleset=R, map="standard",
                                     shrink_every_n_turns=7), T, G),
        ("royale_empty", GameConfig(num_snakes=3, ruleset=R, map="empty",
                                    shrink_every_n_turns=5), T, G),
        ("royale_7x7", GameConfig(width=7, height=7, num_snakes=4, ruleset=R, map="royale",
                                  shrink_every_n_turns=4), T, G),
        ("royale_9x7_1", GameConfig(width=9, height=7, num_snakes=2, ruleset=R, map="royale",
                                    shrink_every_n_turns=1), T, G),
        ("royale_max", dataclasses.replace(m("royale"), max_turns=60), T, G),
        ("duel_max1", dataclasses.replace(m("duel"), max_turns=1), T, G),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/naz-golden")
    ap.add_argument("--only", default="trans,dist")
    ap.add_argument("--configs", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    parts = set(args.only.split(","))
    names = set(filter(None, args.configs.split(",")))
    root = os.path.join(args.out, "env_review")
    for k, (name, game, target, games) in enumerate(configs()):
        if names and name not in names:
            continue
        if "trans" in parts:
            eg.gen_trans(name, game, max(1, int(target * args.scale)),
                         os.path.join(root, "trans", name), args.seed * 1000 + k)
        if "dist" in parts and games:
            eg.gen_dist(name, game, max(64, int(games * args.scale)),
                        os.path.join(root, "dist"), args.seed * 1000 + 500 + k)


if __name__ == "__main__":
    main()
