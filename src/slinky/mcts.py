"""Simultaneous-move Monte Carlo tree search (SM-MCTS) for Battlesnake.

Every snake moves at once, so the search tree is a *stacked matrix game*: each
node has one child per **joint** action (``4**N`` of them, 16 in a duel), and
each player keeps **decoupled** statistics over its own four moves. One search
serves every seat: :func:`search` returns each player's own recommendation.
The design follows the implementation spec in
``docs/research/simultaneous_move_mcts.md``.

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
tree (checked in the compiled HLO), and the time per simulation grows only
slowly with ``M`` (cache misses, deeper descents).

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
   ``env.win_loss_reward``: dead -1, sole survivor +1, snakes dying together on
   the last turn 0; a truncated game is 0 for every living snake). Otherwise
   optional rollouts (``rollout_steps`` turns of ``rollout_policy``) and then
   :func:`slinky.heuristic.evaluate` (``leaf="heuristic"``, in ``[-0.99,
   0.99]`` so exact outcomes always dominate) or 0 for living snakes
   (``leaf="none"``).
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
  moves come first (priority ``1e6 + U(0, 1)``), and ``tie_noise * U(0, 1)``
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

**Final move** (``final``). ``"max"`` (default) plays each player's most
visited move (DUCT) or the largest average-strategy move (RM), breaking ties
by mean value. ``"sample"`` draws from the visit distribution (DUCT) or the
average strategy (RM), which is what the convergence results are about.
DUCT(max) beat DUCT(mix) 58% in Tron (Lanctot et al. 2013).

**Defaults and why** (measured in the 11x11 duel against the heuristic snake
of :mod:`slinky.heuristic`, with 95% confidence intervals; see
:class:`MCTSConfig`):

* DUCT with heuristic leaves and no rollouts, as the research recommends:
  evaluation functions beat random playouts in Tron for every sampling method
  (Bosansky et al. 2016).
* ``exploration=0.25`` rather than UCB1's ``sqrt(2)``. The heuristic's values
  are compressed (the median ``|value|`` over states of heuristic play is
  0.25, so 0.12 from the middle of UCB's ``[0, 1]`` scale), and a large ``C``
  spreads the visits almost uniformly. At 64 simulations ``C = 1.4`` scored 0.49 +- 0.03
  and ``C = 0.25`` 0.54 +- 0.03 (256 games each).
* ``spawn_food=False``: the same strength per simulation as sampled spawns
  (0.594 +- 0.043 against 0.590 +- 0.046 at 64 simulations) at about 1.5x the
  simulations per second, since the spawn's random draw costs more than the
  rest of the rules step.
* ``final="max"`` (DUCT(max) beat DUCT(mix) 58% in Tron; Lanctot et al. 2013).
* ``max_depth=32``: never reached in practice (the deepest node at 1024
  simulations was 11 moves down), but it bounds the path buffer.

**Limits.** The children table has ``4**N`` entries per node, so ``N <= 4``
(256 joint actions; more snakes raise ``ValueError``). The spec was written for
duels; 3 and 4 snakes run (values are per player, not constant-sum), but they
have not been tuned or benchmarked. The tree is rebuilt every turn, since a
``Policy`` keeps no state between turns. Food spawns inside the tree are either
ignored (the default) or sampled once per node, so a node never stands for an
average over spawns.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from slinky import heuristic, rules
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy
from slinky.types import NUM_ACTIONS, GameConfig, State, TimeStep

MAX_SNAKES = 4  # the joint-action children table has 4**N entries per node
_UNVISITED = 1e6  # selection priority of unvisited moves (finite, so noise breaks ties)
_PHI = 0.6180339887498949  # golden-ratio offset: fresh-looking noise at every depth
_SELECTIONS = ("duct", "rm")
_LEAVES = ("heuristic", "none")
_ROLLOUT_POLICIES = ("random", "heuristic")
_FINALS = ("max", "sample")


@dataclasses.dataclass(frozen=True)
class MCTSConfig:
    """Static (hashable) search settings; changing one re-specializes jitted code.

    Attributes:
      num_simulations: simulations per search; the tree has one more node.
      selection: ``"duct"`` (decoupled UCT) or ``"rm"`` (regret matching).
      exploration: DUCT's UCB1 constant ``C``, on the ``[0, 1]`` value scale.
      ucb1_tuned: DUCT uses UCB1-Tuned's variance bound instead of ``C``.
      tie_noise: scale of the ``U(0, 1)`` noise added to DUCT scores.
      rm_gamma: RM's exploration mix (0.1-0.3 in the literature).
      leaf: ``"heuristic"`` (:func:`slinky.heuristic.evaluate`) or ``"none"``
        (0 for every living snake) at non-terminal leaves.
      rollout_steps: turns played from each new node before the leaf
        evaluation (0 = none); a rollout stops early if the game ends.
      rollout_policy: ``"random"`` (uniform over ``env.action_mask``) or
        ``"heuristic"`` (:func:`slinky.heuristic.heuristic_policy`, about 25x
        the cost of a random step).
      weights: the heuristic's weights (leaf evaluator and rollout policy).
      max_depth: the deepest a simulation descends (edges from the root). A
        selection that reaches it backs up that node's stored leaf value.
      final: ``"max"`` or ``"sample"`` (see the module docstring).
      draw_value: value of a mutual elimination inside the search (0 matches
        ``env.win_loss_reward``; negative values are "contempt" for draws).
      spawn_food: transitions inside the tree (and rollouts) sample the map's
        food spawn with ``env.step``. If False they run ``rules.rules_step``
        with no spawn, a deterministic model of the game (Schier &
        Wustenbecker 2019) that skips the spawn's random draw.
    """

    num_simulations: int = 128
    selection: str = "duct"
    exploration: float = 0.25
    ucb1_tuned: bool = False
    tie_noise: float = 0.01
    rm_gamma: float = 0.2
    leaf: str = "heuristic"
    rollout_steps: int = 0
    rollout_policy: str = "random"
    weights: heuristic.Weights = heuristic.DEFAULT_WEIGHTS
    max_depth: int = 32
    final: str = "max"
    draw_value: float = 0.0
    spawn_food: bool = False

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must be >= 0")
        for name, allowed in (
            ("selection", _SELECTIONS),
            ("leaf", _LEAVES),
            ("rollout_policy", _ROLLOUT_POLICIES),
            ("final", _FINALS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")


class SearchOutput(NamedTuple):
    """Result of one search, from every player's own point of view (``N`` players)."""

    action: jax.Array  # int32[N]: each player's recommended move (see ``final``)
    policy: jax.Array  # float32[N, 4]: visit distribution (DUCT) or average strategy (RM)
    visits: jax.Array  # int32[N, 4]: root visits of each player's own moves
    q: jax.Array  # float32[N, 4]: mean value of each own move at the root (0 if unvisited)
    value: jax.Array  # float32[N]: root value estimate in [-1, 1]
    nodes_used: jax.Array  # int32[]: expanded nodes, root included
    depth: jax.Array  # int32[]: depth of the deepest node


class Tree(NamedTuple):
    """One game's search tree; ``M`` nodes, ``N`` players, ``J = 4**N`` joint actions.

    The RM fields are ``None`` under DUCT, and ``value_sq`` is ``None``
    unless ``ucb1_tuned``.
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


class _Descent(NamedTuple):
    """Carry of the selection loop (small, since a vmapped while_loop selects all of it)."""

    node: jax.Array  # int32: current node
    depth: jax.Array  # int32: edges taken so far
    path_node: jax.Array  # int32[D]: node at each level of the path
    path_joint: jax.Array  # int32[D]: joint action taken there
    expand: jax.Array  # bool: stopped at an unexpanded joint action of ``node``
    stored: jax.Array  # float32[N]: ``tree.value[node]`` (backed up when not expanding)
    active: jax.Array  # bool


# --- Helpers -------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _search_env(config: GameConfig) -> BattlesnakeEnv:
    """An observation-free env for ``config`` (observations would dominate the cost)."""
    return BattlesnakeEnv(config, obs=None)


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


def _leaf_value(
    key: jax.Array, state: State, mask: jax.Array, env: BattlesnakeEnv, config: MCTSConfig
) -> jax.Array:
    """float32[N] value of a newly expanded node (``mask`` is its ``env.action_mask``)."""
    if config.rollout_steps > 0:
        state = _rollout(key, state, mask, env, config)
    exact = heuristic.terminal_values(state, env.config, config.draw_value)
    if config.leaf == "heuristic":
        estimate = heuristic.evaluate(state, env.config, config.weights)
    else:
        estimate = jnp.where(state.alive, 0.0, -1.0)
    # ``done`` covers truncation too (living snakes get 0 there, as a draw).
    return jnp.where(state.done, exact, estimate).astype(jnp.float32)


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
    nf = n.astype(jnp.float32)
    n1 = jnp.maximum(nf, 1.0)
    log_n = jnp.log(jnp.maximum(jnp.sum(nf, axis=-1, keepdims=True), 1.0))
    q = tree.value_sum[node] / n1  # [-1, 1]
    if config.ucb1_tuned:
        var = jnp.maximum(tree.value_sq[node] / n1 - q * q, 0.0) * 0.25  # on the [0, 1] scale
        bonus = jnp.sqrt(log_n / n1 * jnp.minimum(0.25, var + jnp.sqrt(2.0 * log_n / n1)))
    else:
        bonus = config.exploration * jnp.sqrt(log_n / n1)
    score = jnp.where(n == 0, _UNVISITED + u, 0.5 * (q + 1.0) + bonus + config.tie_noise * u)
    return jnp.argmax(jnp.where(tree.legal[node], score, -jnp.inf), axis=-1).astype(jnp.int32)


def _rm_joint(
    tree: Tree, node: jax.Array, u: jax.Array, config: MCTSConfig, digits: jax.Array
) -> jax.Array:
    """int32[] the regret-matching joint action at ``node``; ``u`` is U(0,1)[N, 4]."""
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
    uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
    probs = (1.0 - config.rm_gamma) * _rm_sigma(tree.regret[node], legal)
    probs = probs + config.rm_gamma * uniform
    cdf = jnp.cumsum(probs, axis=-1)
    r = jnp.clip(u[:, 0], 1e-6, 1.0 - 1e-6)[:, None] * cdf[:, -1:]
    moves = jnp.minimum(jnp.sum(cdf < r, axis=-1), NUM_ACTIONS - 1).astype(jnp.int32)
    return jnp.where(n_fresh > 0, pick_fresh, jnp.sum(moves * powers))


# --- Search ------------------------------------------------------------------------------


def _init_tree(
    state: State, env: BattlesnakeEnv, config: MCTSConfig, num_nodes: int
) -> Tree:
    n = env.config.num_snakes
    j = NUM_ACTIONS**n
    mask = env.action_mask(state)
    if config.leaf == "heuristic":
        estimate = heuristic.evaluate(state, env.config, config.weights)
    else:
        estimate = jnp.where(state.alive, 0.0, -1.0)
    exact = heuristic.terminal_values(state, env.config, config.draw_value)
    value = jnp.where(state.done, exact, estimate)

    def stack(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.zeros((num_nodes, *x.shape), x.dtype).at[0].set(x)

    def zeros(*shape: int, dtype=jnp.float32) -> jax.Array:
        return jnp.zeros((num_nodes, *shape), dtype)

    rm = config.selection == "rm"
    return Tree(
        state=jax.tree.map(stack, state),
        children=jnp.full((num_nodes, j), -1, jnp.int32),
        legal=stack(_legal(state, mask, env.config)),
        terminal=stack(state.done),
        value=stack(value.astype(jnp.float32)),
        visits=zeros(n, NUM_ACTIONS, dtype=jnp.int32),
        value_sum=zeros(n, NUM_ACTIONS),
        value_sq=zeros(n, NUM_ACTIONS) if config.ucb1_tuned else None,
        regret=zeros(n, NUM_ACTIONS) if rm else None,
        strategy_sum=zeros(n, NUM_ACTIONS) if rm else None,
        joint_visits=zeros(j, dtype=jnp.int32) if rm else None,
        joint_value=zeros(j, n) if rm else None,
    )


def _descend(
    tree: Tree, u: jax.Array, config: MCTSConfig, digits: jax.Array, max_depth: int
) -> _Descent:
    """Selection: walk down from the root, recording the path (see the module docstring)."""
    n = digits.shape[1]
    powers = NUM_ACTIONS ** jnp.arange(n, dtype=jnp.int32)

    def body(c: _Descent) -> _Descent:
        u_d = jnp.mod(u + c.depth.astype(jnp.float32) * _PHI, 1.0)
        if config.selection == "duct":
            joint = jnp.sum(_duct_moves(tree, c.node, u_d, config) * powers)
        else:
            joint = _rm_joint(tree, c.node, u_d, config, digits)
        child = tree.children[c.node, joint]
        depth = c.depth + 1
        expand = child < 0
        node = jnp.where(expand, c.node, child)
        stop = expand | tree.terminal[node] | (depth >= max_depth)
        return _Descent(
            node=node,
            depth=depth,
            path_node=c.path_node.at[c.depth].set(c.node),
            path_joint=c.path_joint.at[c.depth].set(joint),
            expand=expand,
            stored=tree.value[node],
            active=~stop,
        )

    zero = jnp.zeros((), jnp.int32)
    init = _Descent(
        node=zero,
        depth=zero,
        path_node=jnp.zeros((max_depth,), jnp.int32),
        path_joint=jnp.zeros((max_depth,), jnp.int32),
        expand=jnp.zeros((), bool),
        stored=tree.value[0],
        active=~tree.terminal[0],
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
    if config.ucb1_tuned:
        tree = tree._replace(value_sq=tree.value_sq.at[at].add(v * v, mode="drop"))
    if config.selection == "rm":
        safe = jnp.where(on_path, path.path_node, 0)  # gathers only; their updates are dropped
        legal = tree.legal[safe]  # [D, N, 4]
        sigma = _rm_sigma(tree.regret[safe], legal)
        # Q_p(b): the joint table's mean for move b against the others' sampled moves.
        powers = NUM_ACTIONS ** jnp.arange(n, dtype=jnp.int32)
        b = jnp.arange(NUM_ACTIONS)[None, None, :]
        cells = path.path_joint[:, None, None] + (b - moves[:, :, None]) * powers[None, :, None]
        nj = tree.joint_visits[safe[:, None, None], cells]  # [D, N, 4]
        xj = tree.joint_value[safe[:, None, None], cells, players[:, :, None]]
        vb = v[:, :, None]
        q = jnp.where(nj > 0, xj / jnp.maximum(nj, 1), vb)
        q = jnp.where(b == moves[:, :, None], vb, q)
        tree = tree._replace(
            regret=tree.regret.at[nodes].add(jnp.where(legal, q - vb, 0.0), mode="drop"),
            strategy_sum=tree.strategy_sum.at[nodes].add(sigma, mode="drop"),
            joint_visits=tree.joint_visits.at[nodes, path.path_joint].add(1, mode="drop"),
            joint_value=tree.joint_value.at[nodes, path.path_joint].add(v, mode="drop"),
        )
    return tree


def search(
    key: jax.Array, state: State, env: BattlesnakeEnv, config: MCTSConfig = MCTSConfig()
) -> SearchOutput:
    """Run SM-MCTS from ``state`` (one unbatched game) for every player at once.

    Works under ``jit`` and ``vmap`` (close over ``env`` and ``config``). Only
    ``env.config`` is used: transitions run on an observation-free env. A
    finished ``state`` is handled: no simulation expands anything, and each
    player gets a valid move (its first legal one) with zero visits.
    """
    game = env.config
    n = game.num_snakes
    if n > MAX_SNAKES:
        raise ValueError(f"MCTS supports at most {MAX_SNAKES} snakes (4**N joint actions)")
    env = env if env.obs_fn is None else _search_env(game)
    num_sims = config.num_simulations
    max_depth = min(config.max_depth, num_sims)
    digits = jnp.asarray(_digits(n))
    k_noise, k_spawn, k_roll, k_final = jax.random.split(key, 4)
    noise = jax.random.uniform(k_noise, (num_sims, n, NUM_ACTIONS))

    def simulate(sim, carry):
        tree, deepest, expanded = carry
        path = _descend(tree, noise[sim], config, digits, max_depth)
        leaf, slot = path.node, sim + 1
        # Expansion is computed unconditionally (under vmap both branches of a
        # cond would run anyway); without one, slot ``sim + 1`` stays unreferenced.
        joint = path.path_joint[jnp.maximum(path.depth - 1, 0)]
        parent = jax.tree.map(lambda x: x[leaf], tree.state)
        child, mask = _transition(
            jax.random.fold_in(k_spawn, slot), parent, digits[joint], env, config
        )
        new_value = _leaf_value(jax.random.fold_in(k_roll, slot), child, mask, env, config)
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
        tree = _backup(tree, path, value, config, digits)
        deepest = jnp.maximum(deepest, jnp.where(path.expand, path.depth, 0))
        return tree, deepest, expanded + path.expand.astype(jnp.int32)

    zero = jnp.zeros((), jnp.int32)
    tree = _init_tree(state, env, config, num_sims + 1)
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
    )


# --- Policy ------------------------------------------------------------------------------


def mcts_policy(
    key: jax.Array, state: State, env: BattlesnakeEnv, config: MCTSConfig = MCTSConfig()
) -> jax.Array:
    """int32[N] every player's own search recommendation (one search serves all seats)."""
    return search(key, state, env, config).action


def mcts(env: BattlesnakeEnv, config: MCTSConfig = MCTSConfig()) -> Policy:
    """SM-MCTS as an ``evaluate.Policy``: ``(key, state, timestep) -> int32[N]``.

    Cached on ``(env, config)``, so repeated calls return the same object and
    hit ``play_match``'s jit cache. Observations are ignored.
    """
    return _mcts(env, config)


@functools.lru_cache(maxsize=64)
def _mcts(env: BattlesnakeEnv, config: MCTSConfig) -> Policy:
    search_env = _search_env(env.config)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return mcts_policy(key, state, search_env, config)

    return policy
