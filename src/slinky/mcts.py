"""Simultaneous-move Monte Carlo tree search (SM-MCTS) for Battlesnake.

Every snake moves at once, so the search tree is a *stacked matrix game*: each
node has one child per **joint** action (``4**N`` of them, 16 in a duel), and
each player keeps **decoupled** statistics over its own four moves. One search
serves every seat: :func:`search` returns each player's own recommendation.
The design follows the survey and implementation spec in
``docs/research/simultaneous_move_mcts.md``. The code departs from that spec
in its value scale, defaults and option names (the spec lists the changes);
where they differ, this docstring and :class:`MCTSConfig` are authoritative.

**Tree layout.** One fixed-capacity tree per game, ``M = num_simulations + 1``
nodes (:class:`Tree`). Node 0 is the root and simulation ``i`` owns slot
``i + 1``, so no allocation counter is needed: if the simulation expands a
node it lands there, otherwise the slot is never referenced. Each node stores
its full :class:`State` (about 0.9 KB of the 1.04 KB per node in an 11x11
duel), so a simulation costs one transition and one leaf evaluation, never a
replay from the root. Per node there are also ``children[4**N]`` (``-1`` = not
expanded), the move mask ``legal[N, 4]``, ``terminal`` and the leaf
``value[N]``, plus the selection statistics. Everything is updated in place
with ``.at[]`` inside a ``fori_loop`` over simulations; XLA never copies the
tree (a test checks the compiled HLO, batched and not; one unbatched game needs
the barrier of :func:`_unbatched_barrier` for that), and the time per
simulation grows only slowly with ``M`` (cache misses, deeper descents).

**One simulation.**

1. *Selection* descends from the root while the node is non-terminal and the
   chosen joint child exists, up to ``max_depth`` edges. The path (node and
   joint action per level) is recorded in a small buffer, so the backup needs
   no pointer chasing: a ``while_loop`` over the tree under ``vmap`` would
   ``select`` the whole tree on every iteration.
2. *Expansion* of one node with the exact rules (head-to-heads, tail rule,
   starvation, hazards, ``max_turns`` truncation). By default
   (``spawn_food=False``) this is ``rules.rules_step`` without the map's food
   spawn, a deterministic model of the game (as in Schier & Wustenbecker 2019).
   With ``spawn_food=True`` it is ``env.step`` on an observation-free env: the
   spawn is drawn once, when the node is expanded, from a key folded with the
   slot number, so the tree is one consistent sample of the chance events.
3. *Leaf evaluation*: the exact outcome if the game is over (as
   ``env.win_loss_reward``: dead -1, sole survivor +1, and ``draw_value`` for
   snakes dying together on the last turn, 0 in the reward but -0.5 by default
   here; a truncated game is 0 for every living snake). Otherwise
   optional rollouts (``rollout_steps`` turns of ``rollout_policy``) and then
   :func:`slinky.heuristic.evaluate` (``leaf="heuristic"``, in ``[-0.99,
   0.99]`` so exact outcomes always dominate), 0 for living snakes
   (``leaf="none"``) or a user-supplied evaluator, such as a value network
   (``leaf="network"``; see *AlphaZero* below).
4. *Backup* of the per-player value vector ``v[N]`` to every node on the path,
   one scatter-add per statistic. Each player's statistics use its own value.

**Move masks.** Selection only considers ``env.action_mask`` moves (those that
do not certainly hit a wall or body). Dead snakes and snakes that certainly
starve get a single action (0): their move cannot change the outcome, and
splitting the joint children on it would waste the budget. A snake with no
safe move keeps the env's all-True row. That is exact: going out of bounds
removes its body before collisions, while crashing into a body keeps it, which
can take an opponent down too.

**Selection rules** (``MCTSConfig.selection``). Values are kept in
``[-1, 1]`` and mapped to ``[0, 1]`` inside the formulas, so ``C`` has its
usual UCB1 meaning.

* ``"duct"`` (default), decoupled UCT. Each player ``p`` independently takes
  ``argmax_a q01 + C * sqrt(ln n / n_a)`` over its legal moves, where ``q01``
  is the mean value of ``a`` mapped to ``[0, 1]``, ``n_a`` its visit count and
  ``n`` the node's visit count. ``ucb1_tuned`` replaces ``C`` with the
  variance bound ``sqrt(min(1/4, var + sqrt(2 ln n / n_a)))``. Unvisited legal
  moves come first, in uniformly random order, and ``tie_noise * U(0, 1)``
  is added to every score so near-ties break at random. Deterministic
  tie-breaking makes DUCT cycle in lock-step (Bosansky et al. 2016). DUCT was
  the most robust variant across nine games (Tak et al. 2014) and strong in
  Tron (Lanctot et al. 2013). It is the simplest to extend to N players.
* ``"rm"``, regret matching, the one alternative the research note
  recommends. Each node also keeps a joint table of visits and value sums
  (``[4**N]``, ``[4**N, N]``) and per-player regrets and strategy sums.
  Unvisited legal joint cells are tried first, uniformly at random. Then each
  player samples from ``(1 - rm_gamma) * sigma + rm_gamma * uniform``, where
  ``sigma`` is proportional to the positive regrets. The backup adds ``sigma``
  to the strategy sum and, for each alternative move ``b``, ``Q(b) - v`` to the
  regret, where ``Q(b)`` is the joint table's mean for ``b`` against the other
  players' sampled moves. RM empirically converges to a Nash equilibrium
  (Lisy et al. 2013); DUCT need not.
* ``"puct"``, decoupled PUCT: AlphaZero's rule, applied by each player to its
  own moves. Player ``p`` takes ``argmax_a q01 + c_puct * P(a) * sqrt(max(N,
  1)) / (1 + n_a) + tie_noise * U(0, 1)`` over its legal moves, where ``P`` is
  its prior at the node (``Tree.prior``) and ``N`` its visits there. An
  unvisited move's ``q01`` is the node's own stored leaf value minus the
  constant ``fpu_reduction``, mapped to ``[0, 1]``: a simple first-play
  urgency (Leela Zero and KataGo scale their reduction with the square root of
  the prior mass already visited, from the node's evaluation or mean value).
  There is no unvisited-first rule: the prior decides which moves are tried,
  so a move with a small prior may get no visit at all. ``ucb1_tuned`` is
  refused here: PUCT has no use for the variance.

**Final move** (``final``). ``"max"`` (default) plays each player's most
visited move (DUCT, PUCT) or the largest average-strategy move (RM), breaking
ties by mean value. ``"sample"`` draws from the visit distribution (DUCT,
PUCT) or the average strategy (RM), which is what the convergence results are
about.
DUCT(max) beat DUCT(mix) 58% in Tron (Lanctot et al. 2013).

**AlphaZero: priors, network leaves and root noise.** Under PUCT each node
stores its priors (``Tree.prior``, ``float32[M, N, 4]``, 32 bytes per node in
a duel; ``None`` under DUCT and RM, which are unchanged), set when it is
expanded: uniform over the legal moves (``prior="uniform"``) or a softmax of
the evaluator's logits over them (``prior="network"``; logits are clipped to
``+-1e30`` first, so a row whose legal logits are all ``-inf`` is uniform, not
NaN). A forced row (dead or starving snake) is 1 on its move. The evaluator
is passed to :func:`search` as
``evaluator(state) -> (value f32[N], logits f32[N, 4])`` and called on ONE
unbatched state, at the root and once per simulation on its new node; a
``vmap`` of the search over B games makes each call one network evaluation
over B states (2B observations in a duel). It is a
closure, so the network's parameters can be traced arguments of the jitted
self-play step, and new parameters need no recompilation. They cannot live
in the hashable ``MCTSConfig``, which is also why :func:`mcts` refuses
configs that need an evaluator: a trainer or a network agent builds its own
``Policy`` around ``search(..., evaluator=...)``. ``leaf="network"`` uses
the evaluator's value (clipped to ``[-1, 1]``) at non-terminal leaves; exact
outcomes still override it, so value targets should be the game's outcome
per snake on the same scale (``heuristic.terminal_values`` with the search's
``draw_value``). The two settings mix freely: network priors with the
heuristic leaf, or a network leaf with uniform priors (or under DUCT).
``root_noise_frac > 0`` mixes ``Dirichlet(root_dirichlet_alpha)`` noise over
each player's legal moves into its root priors, for self-play exploration;
its key is folded from the search key, so everything else in the search
draws the same randomness as without it. The noise is normalized in log
space (a softmax of log-Gamma draws): with a small alpha every Gamma draw of a
row can underflow float32 (at alpha 0.01, 5% of rows then summed to less than
0.999). ``SearchOutput.policy`` (each player's share of its root visits) is
the policy target, and ``SearchOutput.root_prior`` reports the root priors
used, after the noise (a Gumbel-style target, which needs the noise-free
prior, should take it from the evaluator).
Self-play should sample its opening moves from ``policy`` outside the
search: ``final`` is static, and sampling every move is weak (see the
``final="max"`` default below).

**Defaults and why.** Measured in the 11x11 duel against the heuristic snake
(:func:`slinky.heuristic.heuristic`), with 95% confidence intervals. Unless
noted, each line is 256 games at 64 simulations, on the same starting
positions, against the final default's 0.594 +- 0.043:

* Heuristic leaves, no rollouts, as the research recommends (evaluation
  functions beat random playouts in Tron for every sampling method; Bosansky
  et al. 2016). Ten random-rollout steps before the heuristic scored 0.113 +-
  0.035 (0.102 +- 0.045 with ``C = 1.4``) at 3.3x the cost per simulation, and
  pure random rollouts (``leaf="none"``, 30 steps) 0.029 +- 0.018, although
  both still beat ``random_legal`` 0.99.
* DUCT. Regret matching (``rm_gamma=0.2``) scored 0.551 +- 0.037 at 1.8x the
  cost per simulation, and UCB1-Tuned 0.533 +- 0.044. (RM was measured while
  its samples below the root still depended on the parent's sample; every
  depth of a descent now draws its own noise.)
* ``exploration=0.25`` rather than UCB1's ``sqrt(2)``. The heuristic's values
  are compressed (the median ``|value|`` over states of heuristic play is
  0.25, so 0.12 from the middle of UCB's ``[0, 1]`` scale), and a large ``C``
  spreads the visits almost uniformly. With a draw worth 0, ``C = 1.4`` scored
  0.490 +- 0.029 and ``C = 0.25`` 0.539 +- 0.030; with contempt, ``C = 0.1``
  and 0.25 scored 0.59 and ``C = 0.5`` 0.525 +- 0.048.
* ``draw_value=-0.5``, contempt for mutual eliminations. With a draw worth 0,
  as in ``env.win_loss_reward``, the search happily trades heads at equal
  length whenever its position looks slightly worse, and about 75% of games
  against the heuristic were draws. With contempt they are decided by play:
  0.539 +- 0.030 rose to 0.590 +- 0.046 at 64 simulations, and 0.596 +- 0.034
  to 0.700 +- 0.049 at 256 (192 games). Against the DQN it made no
  difference (0.961 and 0.969 +- 0.03 at 64, 128 games). A draw still beats a
  loss, and ``draw_value=0`` gives the pure win/loss objective.
* ``spawn_food=False``: the same strength per simulation as sampled spawns
  (0.594 +- 0.043 against 0.590 +- 0.046) at 1.4x the simulations per second,
  because the spawn's random draw costs more than the rest of the rules step.
* ``final="max"``. Sampling the move from the visit counts scored 0.127 +-
  0.035 (from RM's average strategy, 0.365 +- 0.044): at small budgets every
  move keeps a few exploratory visits, and over a hundred turns the sampled
  blunders add up.
* ``tie_noise=0.01``: without it, 0.561 +- 0.046 (within noise).
* ``max_depth=32``: never reached in practice (the deepest node at 1024
  simulations was 10-11 moves down), but it bounds the path buffer.
* PUCT's ``c_puct=0.5`` and ``fpu_reduction=0.2`` were measured without a
  network (uniform priors, heuristic leaves), 512 games at 64 and 256
  simulations against DUCT's 0.603 and 0.715 (+- 0.04): ``c_puct`` 0.1
  scored 0.471 and 0.692, 0.25 0.591 and 0.725, 0.5 0.595 and 0.692, and
  AlphaZero's 1.25 only 0.500 and 0.599 (2.5: 0.426 and 0.527). As for
  DUCT's ``C``, the heuristic's compressed values want little exploration.
  At ``c_puct=0.5``, ``fpu_reduction`` 0 scored 0.585 and 0.695 and 0.5
  scored 0.561 (64 simulations), within noise of 0.2. With network priors
  and a value trained on outcomes both are untuned.
* No root noise (``root_noise_frac=0``): it is for self-play only (AlphaZero
  used 0.25). ``root_dirichlet_alpha=0.5`` is untuned. A snake has 2-3 legal
  moves; AlphaZero's alphas (0.3 in chess, 0.03 in Go) are about 10 over the
  typical number of moves, which would give 3-5 here, nearly flat noise.

**Speed** (4-core CPU, ``jit(vmap(search))``, default config): about 5 us per
simulation at a batch of 256 games (3.9 at 16 simulations, 6.0 at 256; 165k to
255k simulations/s in all) and 6 us at a batch of 64 (64 simulations). The
heuristic leaf is about 1.5 us of that, and the tree takes 1.04 KB per node
(0.27 MB per game at 256 simulations). One unbatched game (``jit(search)`` on
one core, the Battlesnake server setting; ``benchmarks/mcts_latency.py``, on
positions from games of mcts-32 against the heuristic, with the other three
cores busy) takes about 17.5-19 us per simulation from 4096 to 32768
simulations (72 ms at 4096, 296 ms at 16384, 626 ms at 32768). So about 26k
simulations fit a 500 ms move at the median latency (24k at p90), and 21k fit
400 ms (18k at p90).

PUCT and network leaves, on one pinned core (``taskset -c 0``, the other
cores idle), ``jit(vmap(search))`` over 64 games at 64 simulations, on live
positions from heuristic games: DUCT takes 4.9 us per simulation and PUCT
with uniform priors 5.0-5.1 (+3-5%, mostly the prior table). With the PPO
baseline's network as the evaluator (``baselines/ppo.py``: both snakes'
egocentric 21x21x13 observations, conv 16-32-32, dense 256, ``tanh`` value)
for priors and leaf values it is 85-89 us, about 18x: the network is about
95% of a simulation. Network priors with heuristic leaves take 94 us, and
root noise costs nothing measurable.

**Limits.** The children table has ``4**N`` entries per node, so ``N <= 4``
(256 joint actions; more snakes raise ``ValueError``). The spec was written for
duels; 3 and 4 snakes run (values are per player, not constant-sum), but they
have not been tuned or benchmarked. The tree is rebuilt every turn, since a
``Policy`` keeps no state between turns. Food spawns inside the tree are either
ignored (the default) or sampled once per node, so a node never stands for an
average over spawns. The evaluator runs on every simulation, also one that
re-selects a terminal node (or stops at ``max_depth``) and expands nothing:
free under ``vmap``, where both branches would run anyway, but one unbatched
search pays the network for those simulations too.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap

from slinky import heuristic, rules
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy
from slinky.types import NUM_ACTIONS, GameConfig, State, TimeStep

MAX_SNAKES = 4  # the joint-action children table has 4**N entries per node
_SELECTIONS = ("duct", "rm", "puct")
_LEAVES = ("heuristic", "none", "network")
_PRIORS = ("uniform", "network")
_ROLLOUT_POLICIES = ("random", "heuristic")
_FINALS = ("max", "sample")
# ``fold_in(key, _ROOT_NOISE_STREAM)`` draws the root's Dirichlet noise. No 4-way split of
# the search key gives this key (under either threefry mode), and the other keys stay as
# they were, so turning the noise on changes nothing else about a search.
_ROOT_NOISE_STREAM = 0x524F4F54

# ``evaluator(state) -> (value f32[N] in [-1, 1], prior_logits f32[N, 4])`` for ONE
# unbatched state (see :func:`search`).
Evaluator = Callable[[State], tuple[jax.Array, jax.Array]]


@dataclasses.dataclass(frozen=True)
class MCTSConfig:
    """Static (hashable) search settings; changing one re-specializes jitted code.

    Attributes:
      num_simulations: simulations per search; the tree has one more node.
      selection: ``"duct"`` (decoupled UCT), ``"rm"`` (regret matching) or
        ``"puct"`` (decoupled PUCT with priors, for AlphaZero).
      exploration: DUCT's UCB1 constant ``C``, on the ``[0, 1]`` value scale.
      ucb1_tuned: DUCT uses UCB1-Tuned's variance bound instead of ``C`` (refused
        with PUCT; ignored by RM).
      tie_noise: scale of the ``U(0, 1)`` noise added to DUCT and PUCT scores.
      rm_gamma: RM's exploration mix (0.1-0.3 in the literature).
      c_puct: PUCT's prior weight, on the ``[0, 1]`` value scale.
      fpu_reduction: PUCT's first-play urgency: an unvisited move is worth the
        node's own stored value minus this (on the ``[-1, 1]`` scale).
      prior: where PUCT's priors come from: ``"uniform"`` over the legal moves
        or ``"network"`` (the evaluator's logits). Needs ``selection="puct"``.
      root_noise_frac: weight of the Dirichlet noise mixed into each player's
        root priors (0 = off; AlphaZero used 0.25). Needs ``selection="puct"``.
      root_dirichlet_alpha: concentration of that noise, per legal move.
      leaf: ``"heuristic"`` (:func:`slinky.heuristic.evaluate`), ``"none"``
        (0 for every living snake) or ``"network"`` (the evaluator's value) at
        non-terminal leaves.
      rollout_steps: turns played from each new node before the leaf
        evaluation (0 = none); a rollout stops early if the game ends.
      rollout_policy: ``"random"`` (uniform over ``env.action_mask``) or
        ``"heuristic"`` (:func:`slinky.heuristic.heuristic_policy`, about 30x
        the cost of a random step).
      weights: the heuristic's weights (leaf evaluator and rollout policy). Its
        policy fields, such as ``contempt``, only matter for heuristic rollouts;
        the search's own value of a draw is ``draw_value``.
      max_depth: the deepest a simulation descends (edges from the root). A
        selection that reaches it backs up that node's stored leaf value.
      final: ``"max"`` or ``"sample"`` (see the module docstring).
      draw_value: value of a mutual elimination inside the search. 0 matches
        ``env.win_loss_reward``; the default -0.5 is "contempt" for draws (see
        the module docstring).
      spawn_food: if True, transitions inside the tree (and rollouts) use
        ``env.step`` and sample the map's food spawn. If False (the default),
        they run ``rules.rules_step`` without a spawn: a deterministic model of
        the game (Schier & Wustenbecker 2019) that skips the costly random draw.
    """

    num_simulations: int = 128
    selection: str = "duct"
    exploration: float = 0.25
    ucb1_tuned: bool = False
    tie_noise: float = 0.01
    rm_gamma: float = 0.2
    c_puct: float = 0.5
    fpu_reduction: float = 0.2
    prior: str = "uniform"
    root_noise_frac: float = 0.0
    root_dirichlet_alpha: float = 0.5
    leaf: str = "heuristic"
    rollout_steps: int = 0
    rollout_policy: str = "random"
    weights: heuristic.Weights = heuristic.DEFAULT_WEIGHTS
    max_depth: int = 32
    final: str = "max"
    draw_value: float = -0.5
    spawn_food: bool = False

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must be >= 0")
        for name in ("exploration", "tie_noise", "c_puct"):
            if not 0.0 <= getattr(self, name) < math.inf:  # also rejects NaN
                raise ValueError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
        for name in ("rm_gamma", "root_noise_frac"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)!r}")
        if not -2.0 <= self.fpu_reduction <= 2.0:
            raise ValueError(f"fpu_reduction must be in [-2, 2], got {self.fpu_reduction!r}")
        if not 0.0 < self.root_dirichlet_alpha < math.inf:
            raise ValueError(
                f"root_dirichlet_alpha must be finite and > 0, got {self.root_dirichlet_alpha!r}"
            )
        if not -1.0 <= self.draw_value <= 1.0:
            raise ValueError(f"draw_value must be in [-1, 1], got {self.draw_value!r}")
        for name, allowed in (
            ("selection", _SELECTIONS),
            ("leaf", _LEAVES),
            ("prior", _PRIORS),
            ("rollout_policy", _ROLLOUT_POLICIES),
            ("final", _FINALS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.selection != "puct" and (self.prior != "uniform" or self.root_noise_frac > 0):
            raise ValueError("prior and root_noise_frac only apply to selection='puct'")
        if self.selection == "puct" and self.ucb1_tuned:
            raise ValueError("ucb1_tuned is DUCT's bound; selection='puct' does not use it")
        if self.leaf == "network" and self.rollout_steps > 0:
            raise ValueError("leaf='network' evaluates the new node itself: no rollout_steps")

    @property
    def uses_evaluator(self) -> bool:
        """The search calls an evaluator (``leaf`` or ``prior`` is ``"network"``)."""
        return self.leaf == "network" or self.prior == "network"


DEFAULT_CONFIG = MCTSConfig()


class SearchOutput(NamedTuple):
    """Result of one search, from every player's own point of view (``N`` players)."""

    action: jax.Array  # int32[N]: each player's recommended move (see ``final``)
    policy: jax.Array  # float32[N, 4]: visit distribution (DUCT) or average strategy (RM)
    visits: jax.Array  # int32[N, 4]: root visits of each player's own moves
    q: jax.Array  # float32[N, 4]: mean value of each own move at the root (0 if unvisited)
    value: jax.Array  # float32[N]: root value estimate in [-1, 1]
    nodes_used: jax.Array  # int32[]: expanded nodes, root included
    depth: jax.Array  # int32[]: depth of the deepest node
    # float32[N, 4]: the root priors PUCT used (after root noise, so not the network's
    # own prior); uniform over the selectable moves under DUCT and RM
    root_prior: jax.Array


class Tree(NamedTuple):
    """One game's search tree; ``M`` nodes, ``N`` players, ``J = 4**N`` joint actions.

    The RM fields are ``None`` unless ``selection="rm"``, ``prior`` is ``None``
    unless ``selection="puct"``, and ``value_sq`` is ``None`` unless DUCT runs
    with ``ucb1_tuned`` (the only rule that reads it).
    """

    state: State  # State stacked to [M, ...]
    children: jax.Array  # int32[M, J]: child slot per joint action, -1 if not expanded
    legal: jax.Array  # bool[M, N, 4]: moves selection may choose
    terminal: jax.Array  # bool[M]: the game is over (or truncated) at this node
    value: jax.Array  # float32[M, N]: leaf value at expansion (exact if terminal)
    visits: jax.Array  # int32[M, N, 4]: per-player visits of own moves
    value_sum: jax.Array  # float32[M, N, 4]: per-player value sums of own moves
    value_sq: jax.Array | None  # float32[M, N, 4]: sums of squared values (UCB1-Tuned)
    regret: jax.Array | None  # float32[M, N, 4]: RM cumulative regrets
    strategy_sum: jax.Array | None  # float32[M, N, 4]: RM sum of played strategies
    joint_visits: jax.Array | None  # int32[M, J]: RM joint-action visit counts
    joint_value: jax.Array | None  # float32[M, J, N]: RM joint-action value sums
    prior: jax.Array | None  # float32[M, N, 4]: PUCT priors over the ``legal`` moves


class _Descent(NamedTuple):
    """Carry of the selection loop (small, since a vmapped while_loop selects all of it)."""

    node: jax.Array  # int32: current node
    depth: jax.Array  # int32: edges taken so far
    path_node: jax.Array  # int32[D]: node at each level of the path
    path_joint: jax.Array  # int32[D]: joint action taken there
    expand: jax.Array  # bool: stopped at an unexpanded joint action of ``node``
    stored: jax.Array  # float32[N]: ``tree.value[node]`` (backed up when not expanding)
    active: jax.Array  # bool
    # RM only (else None): what the backup needs from each node on the path,
    # gathered here because gathering it in the backup makes XLA copy the tables.
    rm_legal: jax.Array | None  # bool[D, N, 4]
    rm_sigma: jax.Array | None  # float32[D, N, 4]: regret-matching strategy
    rm_q: jax.Array | None  # float32[D, N, 4]: joint-table mean of each alternative move
    rm_seen: jax.Array | None  # bool[D, N, 4]: that joint cell has been visited


# --- Helpers -------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _search_env(config: GameConfig) -> BattlesnakeEnv:
    """An observation-free env for ``config`` (observations would dominate the cost)."""
    return BattlesnakeEnv(config, obs=None)


def _check_game(config: GameConfig) -> None:
    if config.num_snakes > MAX_SNAKES:
        raise ValueError(f"MCTS supports at most {MAX_SNAKES} snakes (4**N joint actions)")


def _digits(n: int) -> np.ndarray:
    """int32[4**n, n]: player ``p``'s move in joint action ``j`` is ``(j // 4**p) % 4``."""
    j = np.arange(NUM_ACTIONS**n)[:, None]
    return ((j // NUM_ACTIONS ** np.arange(n)[None, :]) % NUM_ACTIONS).astype(np.int32)


def _legal(state: State, mask: jax.Array, config: GameConfig) -> jax.Array:
    """bool[N, 4] moves selection may choose: ``mask``, with one move for dead/starving snakes."""
    fixed = ~state.alive | rules.certainly_starving(state, config)
    return jnp.where(fixed[:, None], np.arange(NUM_ACTIONS) == 0, mask)


def _transition(
    key: jax.Array, state: State, actions: jax.Array, env: BattlesnakeEnv, config: MCTSConfig
) -> tuple[State, jax.Array]:
    """(next state, its ``env.action_mask``): ``env.step``, or the rules alone (no spawn)."""
    if config.spawn_food:
        nxt, ts = env.step(key, state, actions)
        return nxt, ts.action_mask
    game = env.config
    nxt = rules.rules_step(state, actions, game)._replace(turn=state.turn + 1)
    done = rules.is_game_over(nxt.alive, game)
    if game.max_turns is not None:
        done = done | (nxt.turn >= game.max_turns)
    nxt = jax.tree.map(lambda old, new: jnp.where(state.done, old, new), state, nxt)
    nxt = nxt._replace(done=state.done | done)
    return nxt, env.action_mask(nxt)


def _static_value(
    state: State, env: BattlesnakeEnv, config: MCTSConfig, net_value: jax.Array | None = None
) -> jax.Array:
    """float32[N]: the exact outcome if the game is over, else the leaf estimate.

    ``net_value`` is the evaluator's value of ``state`` (needed for ``leaf="network"``).
    """
    exact = heuristic.terminal_values(state, env.config, config.draw_value)
    if config.leaf == "heuristic":
        estimate = heuristic.evaluate(state, env.config, config.weights)
    elif config.leaf == "network":
        estimate = jnp.where(state.alive, jnp.clip(net_value, -1.0, 1.0), -1.0)
    else:
        estimate = jnp.where(state.alive, 0.0, -1.0)
    # ``done`` covers truncation too (living snakes get 0 there, as a draw).
    return jnp.where(state.done, exact, estimate).astype(jnp.float32)


def _check_evaluator(config: MCTSConfig, evaluator: Evaluator | None) -> None:
    if config.uses_evaluator and evaluator is None:
        raise ValueError(
            f"leaf={config.leaf!r}, prior={config.prior!r} needs an evaluator: "
            "search(key, state, env, config, evaluator=fn)"
        )
    if evaluator is not None and not config.uses_evaluator:
        raise ValueError(
            "an evaluator was given but the config does not use it: set leaf='network' "
            "and/or prior='network' (with selection='puct')"
        )


def _evaluate(
    evaluator: Evaluator | None, state: State, game: GameConfig
) -> tuple[jax.Array, jax.Array]:
    """(value float32[N], prior logits float32[N, 4]) from the user's evaluator."""
    if evaluator is None:
        raise ValueError("this MCTSConfig needs an evaluator (leaf or prior is 'network')")
    value, logits = evaluator(state)
    value, logits = jnp.asarray(value, jnp.float32), jnp.asarray(logits, jnp.float32)
    n = game.num_snakes
    if value.shape != (n,) or logits.shape != (n, NUM_ACTIONS):
        raise ValueError(
            f"the evaluator must return (value f32[{n}], prior logits f32[{n}, {NUM_ACTIONS}]) "
            f"for one unbatched state, got shapes {value.shape} and {logits.shape}"
        )
    return value, logits


def _leaf_value(
    key: jax.Array,
    state: State,
    mask: jax.Array,
    env: BattlesnakeEnv,
    config: MCTSConfig,
    evaluator: Evaluator | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """(float32[N] value, prior logits float32[N, 4] or None) of a newly expanded node.

    ``mask`` is its ``env.action_mask``. The logits are the evaluator's (None if
    the config does not call it).
    """
    net_value = logits = None
    if config.uses_evaluator:
        net_value, logits = _evaluate(evaluator, state, env.config)
    if config.rollout_steps > 0:
        state = _rollout(key, state, mask, env, config)
    return _static_value(state, env, config, net_value), logits


def _node_prior(logits: jax.Array | None, legal: jax.Array, config: MCTSConfig) -> jax.Array:
    """float32[N, 4] PUCT priors over the ``legal`` moves (exactly 0 elsewhere).

    A masked softmax of the evaluator's logits (``prior="network"``) or uniform.
    A forced row (one legal move) is 1 on that move either way. The logits are
    clipped to ``+-1e30`` first (finite logits are unchanged), so a row whose
    legal logits are all ``-inf`` (or several ``+inf``) is uniform over them
    rather than NaN; a NaN logit still propagates.
    """
    if config.prior == "uniform":
        return legal / jnp.sum(legal, axis=-1, keepdims=True)
    logits = jnp.clip(logits, -1e30, 1e30)
    return jax.nn.softmax(jnp.where(legal, logits, -jnp.inf), axis=-1)


def _noisy_root_prior(
    key: jax.Array, prior: jax.Array, legal: jax.Array, config: MCTSConfig
) -> jax.Array:
    """``(1 - frac) * prior + frac * Dirichlet(alpha)`` over each player's legal moves.

    A Dirichlet over a subset of the moves is the normalized independent
    ``Gamma(alpha)`` draws of that subset, so the noise is 0 on illegal moves and
    a forced row keeps its 1. It is normalized in log space, a softmax of
    ``loggamma`` draws (finite for every alpha): below alpha 0.1 all the Gamma
    draws of a row can underflow float32, and dividing by their sum then left
    rows that did not sum to 1 (5% at alpha 0.01).
    """
    log_g = jax.random.loggamma(key, config.root_dirichlet_alpha, prior.shape)
    noise = jax.nn.softmax(jnp.where(legal, log_g, -jnp.inf), axis=-1)
    return (1.0 - config.root_noise_frac) * prior + config.root_noise_frac * noise


def _rollout(
    key: jax.Array, state: State, mask: jax.Array, env: BattlesnakeEnv, config: MCTSConfig
) -> State:
    """Play up to ``rollout_steps`` turns of the rollout policy (stops when the game ends)."""

    def cond(carry):
        i, s, _ = carry
        return (i < config.rollout_steps) & ~s.done

    def body(carry):
        i, s, m = carry
        k_act, k_step = jax.random.split(jax.random.fold_in(key, i))
        if config.rollout_policy == "random":
            acts = jax.random.categorical(k_act, jnp.where(m, 0.0, -jnp.inf), axis=-1)
        else:
            acts = heuristic.heuristic_policy(k_act, s, env, config.weights)
        s, m = _transition(k_step, s, acts.astype(jnp.int32), env, config)
        return i + 1, s, m

    _, state, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), state, mask))
    return state


def _depth_noise(bits: jax.Array, depth: jax.Array) -> jax.Array:
    """float32 U(0, 1) noise for one depth of a descent, from the simulation's uint32 ``bits``.

    A SplitMix-style stream: ``bits + depth * 0x9E3779B9`` (the golden-ratio
    step) through a full-avalanche integer hash (Wellons' lowbias32), so each
    depth gets independent-looking noise. A ``jax.random`` draw per depth made
    the whole search about 10% slower.
    """
    x = bits + depth.astype(jnp.uint32) * jnp.uint32(0x9E3779B9)
    x = (x ^ (x >> 16)) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return (x >> 8).astype(jnp.float32) * (1.0 / (1 << 24))


@custom_vmap
def _unbatched_barrier(x: State) -> State:
    """``jax.lax.optimization_barrier(x)``, except under a ``vmap`` of two or more games.

    Applied to the parent's state, just read from ``tree.state``. Without it,
    XLA compiles one unbatched game (and a ``vmap`` of size 1) by fusing that
    read, a dynamic slice, into every fusion that computes the child, including
    the dynamic-update-slices that store the child in the other ``tree.state``
    arrays. It cannot prove that the slot written is not the node read, so it
    copies every state array (about 1 MB at 1024 simulations) on every
    simulation, and the search became quadratic in ``num_simulations`` (663 us
    per simulation at 4096). The barrier materializes the parent first, so the
    stores are in place. With two or more games the read is a gather, the write
    a scatter, and XLA already updates in place; the barrier there would only
    change the fusion of the batched program, so the ``vmap`` rule drops it.
    """
    return jax.lax.optimization_barrier(x)


@_unbatched_barrier.def_vmap
def _unbatched_barrier_vmap(axis_size: int, in_batched: list, x: State) -> tuple[State, State]:
    (batched,) = in_batched
    if axis_size > 1:
        return x, batched
    # A vmap of size 1 compiles like the unbatched program: leave the decision to the
    # enclosing level (the barrier itself if there is none).
    x = jax.tree.map(lambda a, b: a[0] if b else a, x, batched)
    x = _unbatched_barrier(x)
    return jax.tree.map(lambda a, b: a[None] if b else a, x, batched), batched


def _uses_variance(config: MCTSConfig) -> bool:
    """Whether the tree keeps ``value_sq``: only DUCT's UCB1-Tuned reads it."""
    return config.selection == "duct" and config.ucb1_tuned


def _rm_sigma(regret: jax.Array, legal: jax.Array) -> jax.Array:
    """float32[..., 4] regret matching: positive regrets normalized; uniform if none."""
    pos = jnp.where(legal, jnp.maximum(regret, 0.0), 0.0)
    total = jnp.sum(pos, axis=-1, keepdims=True)
    uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
    return jnp.where(total > 0, pos / jnp.maximum(total, 1e-30), uniform)


# --- Selection rules -------------------------------------------------------------------


def _duct_moves(tree: Tree, node: jax.Array, u: jax.Array, config: MCTSConfig) -> jax.Array:
    """int32[N] each player's UCB1 (or UCB1-Tuned) move at ``node``; ``u`` is U(0,1)[N, 4]."""
    n = tree.visits[node]
    legal = tree.legal[node]
    nf = n.astype(jnp.float32)
    n1 = jnp.maximum(nf, 1.0)
    log_n = jnp.log(jnp.maximum(jnp.sum(nf, axis=-1, keepdims=True), 1.0))
    q = tree.value_sum[node] / n1  # [-1, 1]
    if config.ucb1_tuned:
        var = jnp.maximum(tree.value_sq[node] / n1 - q * q, 0.0) * 0.25  # on the [0, 1] scale
        bonus = jnp.sqrt(log_n / n1 * jnp.minimum(0.25, var + jnp.sqrt(2.0 * log_n / n1)))
    else:
        bonus = config.exploration * jnp.sqrt(log_n / n1)
    score = jnp.where(legal, 0.5 * (q + 1.0) + bonus + config.tie_noise * u, -jnp.inf)
    # Unvisited legal moves come first, ranked by ``u`` alone (a uniformly random order;
    # a large constant plus ``u`` would round to index order in float32).
    fresh = legal & (n == 0)
    score = jnp.where(jnp.any(fresh, axis=-1, keepdims=True), jnp.where(fresh, u, -jnp.inf), score)
    return jnp.argmax(score, axis=-1).astype(jnp.int32)


def _puct_moves(tree: Tree, node: jax.Array, u: jax.Array, config: MCTSConfig) -> jax.Array:
    """int32[N] each player's decoupled PUCT move at ``node``; ``u`` is U(0,1)[N, 4].

    ``argmax_a q01(a) + c_puct * P(a) * sqrt(max(N, 1)) / (1 + n_a) + tie_noise * u``
    over the legal moves, where ``N`` is the player's visits at ``node`` and an
    unvisited move's ``q01`` is the node's stored value minus ``fpu_reduction``
    (first-play urgency), mapped to ``[0, 1]``. No unvisited-first rule: the
    prior decides which moves are tried.
    """
    n = tree.visits[node]
    nf = n.astype(jnp.float32)
    q = tree.value_sum[node] / jnp.maximum(nf, 1.0)  # [-1, 1]
    q = jnp.where(n > 0, q, tree.value[node][:, None] - config.fpu_reduction)
    sqrt_n = jnp.sqrt(jnp.maximum(jnp.sum(nf, axis=-1, keepdims=True), 1.0))
    bonus = config.c_puct * tree.prior[node] * sqrt_n / (1.0 + nf)
    score = 0.5 * (q + 1.0) + bonus + config.tie_noise * u
    return jnp.argmax(jnp.where(tree.legal[node], score, -jnp.inf), axis=-1).astype(jnp.int32)


def _rm_joint(
    tree: Tree, node: jax.Array, u: jax.Array, config: MCTSConfig, digits: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Regret matching at ``node``; ``u`` is U(0,1)[N, 4].

    Returns the joint action and, for the backup, ``legal``, ``sigma``, and the
    joint table's mean (and whether it exists) for each player's alternative
    moves against the other players' chosen moves (all ``[N, 4]``).
    """
    n = digits.shape[1]
    legal = tree.legal[node]  # [N, 4]
    powers = NUM_ACTIONS ** jnp.arange(n, dtype=jnp.int32)
    # Joint cells whose every component is legal, and the unvisited ones among them.
    joint_legal = jnp.all(legal[jnp.arange(n)[None, :], digits], axis=-1)  # [J]
    fresh = joint_legal & (tree.joint_visits[node] == 0)
    n_fresh = jnp.sum(fresh, dtype=jnp.int32)
    k = jnp.minimum((u[0, 1] * n_fresh).astype(jnp.int32), n_fresh - 1)
    pick_fresh = jnp.argmax(fresh & (jnp.cumsum(fresh) == k + 1)).astype(jnp.int32)
    # Otherwise each player samples from (1 - gamma) * sigma + gamma * uniform.
    sigma = _rm_sigma(tree.regret[node], legal)
    uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
    probs = (1.0 - config.rm_gamma) * sigma + config.rm_gamma * uniform
    cdf = jnp.cumsum(probs, axis=-1)
    r = jnp.clip(u[:, 0], 1e-6, 1.0 - 1e-6)[:, None] * cdf[:, -1:]
    moves = jnp.minimum(jnp.sum(cdf < r, axis=-1), NUM_ACTIONS - 1).astype(jnp.int32)
    joint = jnp.where(n_fresh > 0, pick_fresh, jnp.sum(moves * powers))
    # Joint cell with player p's move replaced by b, for every (p, b).
    own = digits[joint]  # [N]
    cells = joint + (jnp.arange(NUM_ACTIONS)[None, :] - own[:, None]) * powers[:, None]
    nj = tree.joint_visits[node, cells]
    xj = tree.joint_value[node, cells, jnp.arange(n)[:, None]]
    return joint, legal, sigma, xj / jnp.maximum(nj, 1), nj > 0


# --- Search ------------------------------------------------------------------------------


def _init_tree(
    state: State,
    env: BattlesnakeEnv,
    config: MCTSConfig,
    num_nodes: int,
    evaluator: Evaluator | None = None,
    noise_key: jax.Array | None = None,
) -> Tree:
    """The tree with only the root; ``noise_key`` draws the root noise (if it is on)."""
    n = env.config.num_snakes
    j = NUM_ACTIONS**n
    mask = env.action_mask(state)
    net_value = logits = None
    if config.uses_evaluator:
        net_value, logits = _evaluate(evaluator, state, env.config)

    def stack(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.zeros((num_nodes, *x.shape), x.dtype).at[0].set(x)

    def zeros(*shape: int, dtype=jnp.float32) -> jax.Array:
        return jnp.zeros((num_nodes, *shape), dtype)

    prior = None
    if config.selection == "puct":
        legal = _legal(state, mask, env.config)
        prior = _node_prior(logits, legal, config)
        if config.root_noise_frac > 0:
            prior = _noisy_root_prior(noise_key, prior, legal, config)
        prior = stack(prior.astype(jnp.float32))

    rm = config.selection == "rm"
    return Tree(
        state=jax.tree.map(stack, state),
        children=jnp.full((num_nodes, j), -1, jnp.int32),
        legal=stack(_legal(state, mask, env.config)),
        terminal=stack(state.done),
        value=stack(_static_value(state, env, config, net_value)),
        visits=zeros(n, NUM_ACTIONS, dtype=jnp.int32),
        value_sum=zeros(n, NUM_ACTIONS),
        value_sq=zeros(n, NUM_ACTIONS) if _uses_variance(config) else None,
        regret=zeros(n, NUM_ACTIONS) if rm else None,
        strategy_sum=zeros(n, NUM_ACTIONS) if rm else None,
        joint_visits=zeros(j, dtype=jnp.int32) if rm else None,
        joint_value=zeros(j, n) if rm else None,
        prior=prior,
    )


def _descend(
    tree: Tree, bits: jax.Array, config: MCTSConfig, digits: jax.Array, max_depth: int
) -> _Descent:
    """Selection: walk down from the root, recording the path (see the module docstring).

    ``bits`` (uint32[N, 4]) is this simulation's randomness. Every depth derives
    its own noise from it (:func:`_depth_noise`), so the moves sampled at a node
    do not depend on those sampled at its parent.
    """
    n = digits.shape[1]
    powers = NUM_ACTIONS ** jnp.arange(n, dtype=jnp.int32)

    rm = config.selection == "rm"

    def body(c: _Descent) -> _Descent:
        u_d = _depth_noise(bits, c.depth)
        rm_fields = {}
        if rm:
            joint, *per_node = _rm_joint(tree, c.node, u_d, config, digits)
            names = ("rm_legal", "rm_sigma", "rm_q", "rm_seen")
            for name, x in zip(names, per_node, strict=True):
                rm_fields[name] = getattr(c, name).at[c.depth].set(x)
        else:
            select = _puct_moves if config.selection == "puct" else _duct_moves
            joint = jnp.sum(select(tree, c.node, u_d, config) * powers)
        child = tree.children[c.node, joint]
        depth = c.depth + 1
        expand = child < 0
        node = jnp.where(expand, c.node, child)
        stop = expand | tree.terminal[node] | (depth >= max_depth)
        return c._replace(
            node=node,
            depth=depth,
            path_node=c.path_node.at[c.depth].set(c.node),
            path_joint=c.path_joint.at[c.depth].set(joint),
            expand=expand,
            stored=tree.value[node],
            active=~stop,
            **rm_fields,
        )

    zero = jnp.zeros((), jnp.int32)
    per_move = (max_depth, n, NUM_ACTIONS)
    init = _Descent(
        node=zero,
        depth=zero,
        path_node=jnp.zeros((max_depth,), jnp.int32),
        path_joint=jnp.zeros((max_depth,), jnp.int32),
        expand=jnp.zeros((), bool),
        stored=tree.value[0],
        active=~tree.terminal[0],
        rm_legal=jnp.zeros(per_move, bool) if rm else None,
        rm_sigma=jnp.zeros(per_move, jnp.float32) if rm else None,
        rm_q=jnp.zeros(per_move, jnp.float32) if rm else None,
        rm_seen=jnp.zeros(per_move, bool) if rm else None,
    )
    return jax.lax.while_loop(lambda c: c.active, body, init)


def _backup(
    tree: Tree, path: _Descent, value: jax.Array, config: MCTSConfig, digits: jax.Array
) -> Tree:
    """Add the sample ``value[N]`` to the statistics of every node on the path."""
    num_nodes = tree.terminal.shape[0]
    d, n = path.path_node.shape[0], digits.shape[1]
    on_path = jnp.arange(d) < path.depth
    nodes = jnp.where(on_path, path.path_node, num_nodes)  # off-path entries are dropped
    moves = digits[path.path_joint]  # [D, N]
    players = jnp.arange(n)[None, :]
    at = (nodes[:, None], players, moves)
    v = jnp.broadcast_to(value[None, :], (d, n))
    tree = tree._replace(
        visits=tree.visits.at[at].add(1, mode="drop"),
        value_sum=tree.value_sum.at[at].add(v, mode="drop"),
    )
    if _uses_variance(config):
        tree = tree._replace(value_sq=tree.value_sq.at[at].add(v * v, mode="drop"))
    if config.selection == "rm":
        # Q_p(b): the joint table's mean for move b against the others' sampled
        # moves (the sample itself for the move played, and if b is unvisited).
        vb = v[:, :, None]
        b = jnp.arange(NUM_ACTIONS)[None, None, :]
        q = jnp.where(path.rm_seen & (b != moves[:, :, None]), path.rm_q, vb)
        regret = jnp.where(path.rm_legal, q - vb, 0.0)
        tree = tree._replace(
            regret=tree.regret.at[nodes].add(regret, mode="drop"),
            strategy_sum=tree.strategy_sum.at[nodes].add(path.rm_sigma, mode="drop"),
            joint_visits=tree.joint_visits.at[nodes, path.path_joint].add(1, mode="drop"),
            joint_value=tree.joint_value.at[nodes, path.path_joint].add(v, mode="drop"),
        )
    return tree


def search(
    key: jax.Array,
    state: State,
    env: BattlesnakeEnv,
    config: MCTSConfig = DEFAULT_CONFIG,
    evaluator: Evaluator | None = None,
) -> SearchOutput:
    """Run SM-MCTS from ``state`` (one unbatched game) for every player at once.

    Works under ``jit`` and ``vmap`` (close over ``env`` and ``config``). Only
    ``env.config`` is used: transitions run on an observation-free env. A
    finished ``state`` is handled: no simulation expands anything, every visit
    count is zero, ``value`` is the exact outcome and each player still gets a
    legal move.

    ``evaluator(state) -> (value f32[N] in [-1, 1], prior_logits f32[N, 4])``
    is required when ``config.leaf`` or ``config.prior`` is ``"network"`` (and
    refused otherwise). It sees ONE unbatched state, like ``search`` itself: a
    ``vmap`` of ``search`` over B games makes each call one batched network
    evaluation over the B states. It is called once at the root and once per
    simulation, on the simulation's new node (also when the simulation expands
    nothing, since under ``vmap`` both branches would run anyway). It may close
    over traced arrays, such as network parameters passed into the jitted
    function, so new parameters need no recompilation. Its value is clipped to
    ``[-1, 1]``; finished states keep their exact values and dead snakes -1.
    """
    game = env.config
    n = game.num_snakes
    _check_game(game)
    _check_evaluator(config, evaluator)
    env = env if env.obs_fn is None else _search_env(game)
    num_sims = config.num_simulations
    max_depth = min(config.max_depth, num_sims)
    digits = jnp.asarray(_digits(n))
    k_noise, k_spawn, k_roll, k_final = jax.random.split(key, 4)
    noise = jax.random.bits(k_noise, (num_sims, n, NUM_ACTIONS), jnp.uint32)

    def simulate(sim, carry):
        tree, deepest, expanded = carry
        path = _descend(tree, noise[sim], config, digits, max_depth)
        leaf, slot = path.node, sim + 1
        # Expansion is computed unconditionally (under vmap both branches of a
        # cond would run anyway); without one, slot ``sim + 1`` stays unreferenced.
        joint = path.path_joint[jnp.maximum(path.depth - 1, 0)]
        parent = _unbatched_barrier(jax.tree.map(lambda x: x[leaf], tree.state))
        child, mask = _transition(
            jax.random.fold_in(k_spawn, slot), parent, digits[joint], env, config
        )
        new_value, new_logits = _leaf_value(
            jax.random.fold_in(k_roll, slot), child, mask, env, config, evaluator
        )
        # Reading ``tree.value[leaf]`` here instead would make XLA copy the array.
        value = jnp.where(path.expand, new_value, path.stored)
        tree = tree._replace(
            state=jax.tree.map(lambda a, x: a.at[slot].set(x), tree.state, child),
            children=tree.children.at[jnp.where(path.expand, leaf, num_sims + 1), joint].set(
                slot, mode="drop"
            ),
            legal=tree.legal.at[slot].set(_legal(child, mask, game)),
            terminal=tree.terminal.at[slot].set(child.done),
            value=tree.value.at[slot].set(new_value),
        )
        if tree.prior is not None:  # (XLA merges the repeated ``_legal``)
            prior = _node_prior(new_logits, _legal(child, mask, game), config)
            tree = tree._replace(prior=tree.prior.at[slot].set(prior.astype(jnp.float32)))
        tree = _backup(tree, path, value, config, digits)
        deepest = jnp.maximum(deepest, jnp.where(path.expand, path.depth, 0))
        return tree, deepest, expanded + path.expand.astype(jnp.int32)

    zero = jnp.zeros((), jnp.int32)
    k_root = jax.random.fold_in(key, _ROOT_NOISE_STREAM) if config.root_noise_frac > 0 else None
    tree = _init_tree(state, env, config, num_sims + 1, evaluator, k_root)
    tree, deepest, expanded = jax.lax.fori_loop(0, num_sims, simulate, (tree, zero, zero))
    return _output(k_final, tree, config, deepest, expanded)


def _output(
    key: jax.Array, tree: Tree, config: MCTSConfig, deepest: jax.Array, expanded: jax.Array
) -> SearchOutput:
    visits, legal = tree.visits[0], tree.legal[0]
    total = jnp.sum(visits, axis=-1)
    q = jnp.where(visits > 0, tree.value_sum[0] / jnp.maximum(visits, 1), 0.0)
    value = jnp.where(
        total > 0, jnp.sum(tree.value_sum[0], axis=-1) / jnp.maximum(total, 1), tree.value[0]
    )
    uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
    if config.selection == "rm":
        weight = jnp.where(legal, tree.strategy_sum[0], 0.0)
    else:
        weight = jnp.where(legal, visits, 0).astype(jnp.float32)
    w_total = jnp.sum(weight, axis=-1, keepdims=True)
    policy = jnp.where(w_total > 0, weight / jnp.maximum(w_total, 1e-30), uniform)
    if config.final == "max":
        top = legal & (policy >= jnp.max(policy, axis=-1, keepdims=True) - 1e-6)
        action = jnp.argmax(jnp.where(top, q, -jnp.inf), axis=-1)
    else:
        action = jax.random.categorical(key, jnp.where(policy > 0, jnp.log(policy), -jnp.inf))
    return SearchOutput(
        action=action.astype(jnp.int32),
        policy=policy.astype(jnp.float32),
        visits=visits,
        q=q.astype(jnp.float32),
        value=value.astype(jnp.float32),
        nodes_used=expanded + 1,
        depth=deepest,
        root_prior=(uniform if tree.prior is None else tree.prior[0]).astype(jnp.float32),
    )


# --- Policy ------------------------------------------------------------------------------


def mcts_policy(
    key: jax.Array,
    state: State,
    env: BattlesnakeEnv,
    config: MCTSConfig = DEFAULT_CONFIG,
    evaluator: Evaluator | None = None,
) -> jax.Array:
    """int32[N] every player's own search recommendation (one search serves all seats)."""
    return search(key, state, env, config, evaluator).action


def mcts(env: BattlesnakeEnv, config: MCTSConfig = DEFAULT_CONFIG) -> Policy:
    """SM-MCTS as an ``evaluate.Policy``: ``(key, state, timestep) -> int32[N]``.

    Cached on ``(env, config)``, so repeated calls return the same object and
    hit ``play_match``'s jit cache. Observations are ignored. Raises
    ``ValueError`` here, not at trace time, for games with more than
    :data:`MAX_SNAKES` snakes, and for configs that need an evaluator: a
    network's parameters cannot be part of a hashable cache key, so a
    network-guided search builds its own ``Policy`` around
    ``search(..., evaluator=...)``.
    """
    _check_game(env.config)
    if config.uses_evaluator:
        raise ValueError(
            f"leaf={config.leaf!r}, prior={config.prior!r} needs an evaluator, which mcts() "
            "cannot cache: call search(key, state, env, config, evaluator=fn) directly"
        )
    return _mcts(env, config)


@functools.lru_cache(maxsize=64)
def _mcts(env: BattlesnakeEnv, config: MCTSConfig) -> Policy:
    search_env = _search_env(env.config)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return mcts_policy(key, state, search_env, config)

    return policy
