"""Battlesnake as a PufferLib 3.0 native environment (the C port in ``battlesnake.h``).

One C game per ``num_snakes`` agent slots: agent ``g * num_snakes + i`` is snake ``i`` of
game ``g``. The rules, maps, rewards and observations are slinky's, exactly
(``tests/test_puffer_env.py`` checks them against slinky and the engine); observations are
``uint8`` vectors laid out as :mod:`puffer.obs_format` describes.

Per step, ``rewards[i]`` and ``terminals[i]`` describe the transition just made, as
PufferLib's trainer expects: ``terminals[i]`` is 1 when snake ``i`` died, when its game
ended (by the rules or by ``max_turns``: PufferLib 3.0 has no truncation signal, so a cut
game counts as terminal), or when the snake was already dead (its slot then holds one-step
episodes with reward 0 until the game ends). A finished game is replaced at once and the
observations are the new game's.

A dead snake's *waiting* slot has no return to learn from (value target 0, advantage about
0 once the trainer has learned the value of the dead observation), but PuffeRL 3.0 cannot
drop rows: the slot is stored, counted as an agent step, and averaged into the losses
(entropy, KL), the advantage normalization and the steps per second. With 3 or more
snakes a fifth to a third of the rows are waiting slots under random legal play; in a duel
the game ends with the first death, so there are none.

Build the extension first: ``.venv-puffer/bin/python puffer/setup.py build_ext --inplace``.
This module does not import JAX.
"""

from __future__ import annotations

import functools
import operator
import sys
import weakref
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `puffer` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium
import numpy as np
import pufferlib

from puffer import binding, obs_format

RULESETS = ("standard", "solo", "wrapped", "constrictor", "wrapped_constrictor", "royale")
MAPS = ("standard", "empty", "royale")
REWARDS = ("win_loss", "placement")
MAX_SNAKES = 16
MAX_DIM = 32

# slinky.modes.mode_config, without importing JAX: (ruleset, map, default snakes, min, max).
# tests/test_puffer_env.py checks this table against slinky.
MODES = {
    "standard": ("standard", "standard", 4, 3, MAX_SNAKES),
    "duel": ("standard", "standard", 2, 2, 2),
    "royale": ("royale", "royale", 4, 2, MAX_SNAKES),
    "constrictor": ("constrictor", "standard", 4, 2, MAX_SNAKES),
}


def check_placement(width: int, height: int, num_snakes: int) -> str:
    """slinky.maps.placement_kind with _random_start_cells' room check (ValueError if rejected)."""
    if num_snakes > MAX_SNAKES:
        raise ValueError(f"the standard maps allow at most {MAX_SNAKES} snakes")
    if width == height:
        if num_snakes > 8 and width < 7:
            raise ValueError("too many snakes for this board size")
        if num_snakes <= 8 and width >= 7:
            return "fixed"
        if width >= 11:
            return "distributed"
    centre = ((width - 1) // 2, (height - 1) // 2)
    room = sum((x + y) % 2 == 0 and (x, y) != centre for x in range(width) for y in range(height))
    if room < num_snakes:
        raise ValueError("not enough room to place snakes")
    return "random"


def resolve_game(
    mode: str | None, num_snakes: int | None, ruleset: str | None, map: str | None
) -> tuple[str, str, int]:
    """``(ruleset, map, num_snakes)`` from a mode or explicit settings, as mode_config would."""
    if mode is None:
        return ruleset or "standard", map or "standard", 2 if num_snakes is None else num_snakes
    if mode not in MODES:
        raise ValueError(f"unknown game mode {mode!r}; expected one of {tuple(MODES)}")
    if ruleset is not None or map is not None:
        raise ValueError("pass either mode or ruleset/map, not both")
    mode_ruleset, mode_map, default, lo, hi = MODES[mode]
    n = default if num_snakes is None else num_snakes
    if not lo <= n <= hi:
        raise ValueError(f"mode {mode!r} cannot have {n} snakes")
    return mode_ruleset, mode_map, n


class Battlesnake(pufferlib.PufferEnv):
    """``num_envs`` Battlesnake games of ``num_snakes`` snakes, stepped in C.

    Args:
      num_envs: number of games; ``num_agents = num_envs * num_snakes``.
      mode: one of ``standard``, ``duel``, ``royale``, ``constrictor``: sets the ruleset,
        map and default snake count as ``slinky.modes.mode_config`` does. It excludes
        ``ruleset`` and ``map``, which otherwise default to ``"standard"``.
      width, height: board size, 1 to 32.
      num_snakes: snakes per game, 1 to 16 (default: the mode's, or 2).
      food_spawn_chance, minimum_food, hazard_damage_per_turn, shrink_every_n_turns: the
        engine's settings (``slinky.types.GameConfig``).
      max_turns: truncate games at this turn (0: no limit, as the engine).
      reward: ``"win_loss"`` or ``"placement"`` (``slinky.env``).
      compute_obs: False skips observations in ``step`` (raw simulation speed only).
      report_interval: return the ``vec_log`` statistics every this many steps (>= 1).
      buf, seed: PufferLib's shared buffers and seed.

    Integer arguments may be numpy integers. ``reset(seed)`` and ``seed`` are taken modulo
    ``2**31 // num_envs`` (game ``g`` is seeded with ``g + seed * num_envs`` in 32 bits in C).
    ``close()`` may be called twice; stepping a closed environment raises ``RuntimeError``.
    """

    def __init__(
        self,
        num_envs: int = 256,
        mode: str | None = None,
        width: int = 11,
        height: int = 11,
        num_snakes: int | None = None,
        ruleset: str | None = None,
        map: str | None = None,
        food_spawn_chance: int = 15,
        minimum_food: int = 1,
        hazard_damage_per_turn: int = 14,
        shrink_every_n_turns: int = 25,
        max_turns: int | None = 0,
        reward: str = "win_loss",
        compute_obs: bool = True,
        report_interval: int = 128,
        buf=None,
        seed: int = 0,
    ) -> None:
        # operator.index: numpy integers pass, floats are a TypeError here and not in C.
        width, height = operator.index(width), operator.index(height)
        food_spawn_chance = operator.index(food_spawn_chance)
        minimum_food = operator.index(minimum_food)
        hazard_damage_per_turn = operator.index(hazard_damage_per_turn)
        shrink_every_n_turns = operator.index(shrink_every_n_turns)
        num_envs, report_interval = operator.index(num_envs), operator.index(report_interval)
        ruleset, map, num_snakes = resolve_game(mode, num_snakes, ruleset, map)
        num_snakes, max_turns = operator.index(num_snakes), operator.index(max_turns or 0)
        if ruleset not in RULESETS:
            raise ValueError(f"unknown ruleset {ruleset!r}; expected one of {RULESETS}")
        if map not in MAPS:
            raise ValueError(f"unknown map {map!r}; expected one of {MAPS}")
        if reward not in REWARDS:
            raise ValueError(f"unknown reward {reward!r}; expected one of {REWARDS}")
        if not (1 <= width <= MAX_DIM and 1 <= height <= MAX_DIM):
            raise ValueError(f"width and height must be in [1, {MAX_DIM}]")
        if not 1 <= num_snakes <= MAX_SNAKES:
            raise ValueError(f"num_snakes must be in [1, {MAX_SNAKES}]")
        if shrink_every_n_turns < 1:
            raise ValueError("shrink_every_n_turns must be at least 1 (the engine rejects 0)")
        if max_turns < 0:
            raise ValueError("max_turns must be >= 0 (0: no limit)")
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if report_interval < 1:
            raise ValueError("report_interval must be >= 1")
        check_placement(width, height, num_snakes)

        self.mode = mode
        self.width, self.height, self.num_snakes = width, height, num_snakes
        self.wrapped = ruleset in ("wrapped", "wrapped_constrictor")
        self.reward = reward
        # slinky.types.GameConfig keyword arguments: GameConfig(**env.game_config).
        self.game_config = dict(
            width=width,
            height=height,
            num_snakes=num_snakes,
            ruleset=ruleset,
            map=map,
            food_spawn_chance=food_spawn_chance,
            minimum_food=minimum_food,
            hazard_damage_per_turn=hazard_damage_per_turn,
            shrink_every_n_turns=shrink_every_n_turns,
            max_turns=max_turns or None,
        )
        self.board_shape = obs_format.board_shape(width, height, self.wrapped)
        self.board_size = obs_format.board_size(width, height, self.wrapped)
        self.obs_size = obs_format.obs_size(width, height, self.wrapped)
        self.compute_obs = bool(compute_obs)
        self.channel_scale = obs_format.channel_scale(width, height)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_envs = num_envs
        self.num_agents = num_envs * num_snakes
        self.report_interval = report_interval
        self.tick = 0
        super().__init__(buf)

        kwargs = dict(self.game_config, max_turns=max_turns, reward=reward)
        kwargs["compute_obs"] = int(bool(compute_obs))
        handles = []
        first_seed = self._reduced_seed(seed) * num_envs
        for g in range(num_envs):
            s = slice(g * num_snakes, (g + 1) * num_snakes)
            handles.append(
                binding.env_init(
                    self.observations[s],
                    self.actions[s],
                    self.rewards[s],
                    self.terminals[s],
                    self.truncations[s],
                    g + first_seed,  # as PufferLib's ocean envs
                    **kwargs,
                )
            )
        self.c_envs = binding.vectorize(*handles)
        self.handles = handles
        # Frees the C games once, at close() or when this object is collected (PufferLib's
        # Serial builds a driver environment and drops it); it holds the pointer, not self.
        self._finalizer = weakref.finalize(self, binding.vec_close, self.c_envs)

    def _reduced_seed(self, seed: int) -> int:
        """``seed`` reduced so that ``g + seed * num_envs`` fits the 32-bit int C computes it in."""
        return operator.index(seed) % (2**31 // self.num_envs)

    def _check_open(self) -> None:
        if self.c_envs is None:
            raise RuntimeError("this Battlesnake environment is closed")

    def reset(self, seed: int | None = None):
        self._check_open()
        self.tick = 0
        binding.vec_reset(self.c_envs, 0 if seed is None else self._reduced_seed(seed))
        return self.observations, []

    def step(self, actions):
        self._check_open()
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        self._check_open()
        binding.vec_render(self.c_envs, 0)

    def close(self):
        self._finalizer()  # frees the C games the first time only
        self.c_envs, self.handles = None, []


def env_creator(name: str = "battlesnake"):
    """A callable building the env, as PufferLib's ocean registry hands out.

    ``battlesnake`` is :class:`Battlesnake`; ``battlesnake_<mode>`` (e.g.
    ``battlesnake_duel``) fixes the mode. A ``puffer_`` prefix is accepted.
    """
    base = name.removeprefix("puffer_")
    if base == "battlesnake":
        return functools.partial(Battlesnake)
    mode = base.removeprefix("battlesnake_")
    if base.startswith("battlesnake_") and mode in MODES:
        return functools.partial(Battlesnake, mode=mode)
    raise ValueError(f"unknown environment {name!r}")


def benchmark(
    env: Battlesnake, seconds: float = 3.0, cache: int = 1024, policy: str = "random"
) -> float:
    """Agent-steps per second of ``env.step`` under a random policy.

    ``policy`` is ``"random"`` (uniformly random int32 actions, pre-generated) or
    ``"random_legal"`` (uniformly random among the moves the observation's mask allows, as
    slinky's ``random_legal_policy``; needs ``compute_obs``). The policy's own cost, a few
    numpy operations per step, is timed too, as slinky's benchmark times its policy.
    """
    import time

    if policy not in ("random", "random_legal"):
        raise ValueError(f"unknown policy {policy!r}; expected 'random' or 'random_legal'")
    if policy == "random_legal" and not env.compute_obs:
        raise ValueError("random_legal reads the mask from the observations: needs compute_obs")
    rng = np.random.default_rng(0)
    mask = env.observations[:, env.board_size : env.board_size + obs_format.MASK_SIZE]
    if policy == "random":
        actions = rng.integers(0, 4, (cache, env.num_agents), dtype=np.int32)

        def act(t: int) -> np.ndarray:
            return actions[t % cache]

    else:
        # Random scores, zeroed on illegal moves: the argmax is uniform over the legal ones
        # (the mask is never all zeros). 64 rows of scores cycle, which is random enough here.
        scores = rng.random((64, env.num_agents, obs_format.MASK_SIZE), dtype=np.float32) + 1e-3
        legal = np.empty((env.num_agents, obs_format.MASK_SIZE), np.float32)
        chosen = np.empty(env.num_agents, np.int32)

        def act(t: int) -> np.ndarray:
            np.multiply(scores[t % 64], mask, out=legal)
            chosen[:] = legal.argmax(axis=1)
            return chosen

    env.reset(0)
    for t in range(16):  # warm up
        env.step(act(t))
    steps, start = 0, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < seconds:
        for _ in range(32):
            env.step(act(steps))
            steps += 1
    return env.num_agents * steps / elapsed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Battlesnake C env throughput")
    parser.add_argument("--mode", default="duel")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--no-obs", action="store_true")
    parser.add_argument("--policy", choices=("random", "random_legal"), default="random")
    args = parser.parse_args()
    env = Battlesnake(num_envs=args.num_envs, mode=args.mode, compute_obs=not args.no_obs)
    sps = benchmark(env, args.seconds, policy=args.policy)
    env.close()
    obs = "without" if args.no_obs else "with"
    print(
        f"{args.mode}: {env.num_agents} agents, {obs} observations, {args.policy}: "
        f"{sps:,.0f} agent-steps/s"
    )


if __name__ == "__main__":
    main()
