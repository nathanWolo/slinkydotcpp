"""The standard map's spawn mask equals the engine's legal-spawn set exactly."""

import numpy as np
import pytest

from slinky.engine_json import state_from_engine
from slinky.maps import spawn_mask
from slinky.types import GameConfig

try:
    from oracle import generate_games, split_games
except ImportError:
    from tests.oracle import generate_games, split_games

pytestmark = pytest.mark.oracle


@pytest.mark.parametrize(
    "flags",
    [
        dict(snakes=4),
        dict(width=7, height=9, snakes=3),
        dict(ruleset="wrapped", snakes=4),
        dict(width=19, height=19, snakes=12),
        dict(width=7, height=7, snakes=1),
    ],
    ids=str,
)
def test_spawn_mask_is_exact(flags, oracle_bin):
    # With minimumFood above the board size, the engine fills every legal cell
    # each turn, so the food it adds is exactly its spawn set.
    config = GameConfig(
        width=flags.get("width", 11),
        height=flags.get("height", 11),
        num_snakes=flags["snakes"],
        ruleset=flags.get("ruleset", "standard"),
    )
    games = split_games(
        generate_games(
            binary=oracle_bin, seed=500, games=20, max_turns=60, minimum_food=10_000, **flags
        )
    )
    trans = [t for g in games for t in g["transitions"]]
    assert trans
    for t in trans:
        mask = np.asarray(spawn_mask(state_from_engine(t["post_rules"], config), config))
        added = {tuple(p) for p in t["post_map"]["food"]} - {
            tuple(p) for p in t["post_rules"]["food"]
        }
        legal = {(int(x), int(y)) for y, x in zip(*np.nonzero(mask), strict=True)}
        assert added == legal, t["post_rules"]
