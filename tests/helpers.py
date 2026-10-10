"""Test helpers shared by test_heuristic.py and test_mcts.py: hand-built boards and rollouts."""

from __future__ import annotations

import functools

import jax

from slinky.engine_json import state_from_engine
from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy
from slinky.types import GameConfig, State

DUEL = GameConfig()


def board(snakes, config=DUEL, food=(), health=None, causes=None, turn=10) -> State:
    """A state from head-first body lists, as in the engine's JSON."""
    health = health or [90] * len(snakes)
    causes = causes or [("", 0)] * len(snakes)
    d = {
        "width": config.width,
        "height": config.height,
        "turn": turn,
        "snakes": [
            {
                "id": f"s{i}",
                "body": [list(c) for c in b],
                "health": health[i],
                "eliminated_cause": causes[i][0],
                "eliminated_on_turn": causes[i][1],
            }
            for i, b in enumerate(snakes)
        ],
        "food": [list(f) for f in food],
        "hazards": [],
    }
    return state_from_engine(d, config)


def both_seats(snakes, **kw):
    """The position with the hero as snake 0, and again as snake 1."""
    health = kw.pop("health", [90, 90])
    return [
        (board(snakes, health=health, **kw), 0),
        (board(snakes[::-1], health=health[::-1], **kw), 1),
    ]


@functools.cache
def env_for(config: GameConfig) -> BattlesnakeEnv:
    return BattlesnakeEnv(config, obs=None)


def rollout_states(config: GameConfig, batch: int = 16, turns: int = 30, seed: int = 0) -> State:
    """A batch of mid-game states from random legal play (autoreset)."""
    env = env_for(config)
    k_reset, k_play = jax.random.split(jax.random.key(seed))
    states, _ = jax.vmap(env.reset)(jax.random.split(k_reset, batch))

    def step(states, key):
        k_pol, k_step = jax.random.split(key)
        acts = jax.vmap(lambda k, s: random_legal_policy(k, s, env))(
            jax.random.split(k_pol, batch), states
        )
        states, _ = jax.vmap(env.step_autoreset)(jax.random.split(k_step, batch), states, acts)
        return states, None

    states, _ = jax.jit(lambda s, k: jax.lax.scan(step, s, jax.random.split(k, turns)))(
        states, k_play
    )
    return states
