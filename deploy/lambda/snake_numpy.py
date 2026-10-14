"""The trained baselines' moves for a Battlesnake API request, in plain numpy (no JAX).

A port of what the trained agents of ``slinky.agents`` compute for one snake,
for hosts where JAX is too heavy (AWS Lambda: JAX costs ~500 MB and seconds of
start-up):

* the request's board as countdown grids (``slinky.engine_json``),
* the egocentric observation (``slinky.observations.egocentric_obs``),
* the legal-move mask (``slinky.rules.action_mask``),
* the networks (:class:`Net`): the policy logits of PPO, REINFORCE and the raw
  AlphaZero network (``az-0``), the DQN's Q-values and Rainbow's expected
  returns (noisy layers at their means), and the greedy move over them
  (``slinky.evaluate.greedy_from_q``: argmax over the legal moves).

It only plays the game the network learned, the 11x11 standard duel; any
other game gets :func:`safe_move`. ``tests/test_snake_numpy.py`` checks it
against the JAX agent move for move.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MOVE_NAMES = ("up", "down", "left", "right")
DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))  # (dx, dy) per move, y up as in the API
SIZE = 11  # the duel board the network was trained on


class Board:
    """A standard-ruleset request as arrays, the requesting snake in seat 0."""

    def __init__(self, req: dict[str, Any]) -> None:
        board, you = req["board"], req["you"]
        self.w, self.h = int(board["width"]), int(board["height"])
        snakes = sorted(board["snakes"], key=lambda s: s["id"] != you["id"])  # stable
        if not snakes or snakes[0]["id"] != you["id"]:
            raise ValueError("'you' is not on the board")
        n = len(snakes)
        self.body = np.zeros((n, self.h, self.w), np.int32)  # countdown: head = length
        self.head = np.zeros((n, 2), np.int64)
        self.length = np.zeros(n, np.int64)
        self.health = np.zeros(n, np.int64)
        for i, s in enumerate(snakes):
            segs = [(int(p["x"]), int(p["y"])) for p in s["body"]]
            if not segs:
                raise ValueError("a snake without a body")
            self.head[i] = segs[0]
            self.length[i] = len(segs)
            self.health[i] = int(s["health"])
            for j, (x, y) in enumerate(segs):
                if 0 <= x < self.w and 0 <= y < self.h:
                    self.body[i, y, x] = max(self.body[i, y, x], len(segs) - j)
        self.food = np.zeros((self.h, self.w), bool)
        for p in board["food"]:
            self.food[int(p["y"]), int(p["x"])] = True
        self.hazard = np.zeros((self.h, self.w), np.float32)
        for p in board.get("hazards", []):
            x, y = int(p["x"]), int(p["y"])
            if 0 <= x < self.w and 0 <= y < self.h:
                self.hazard[y, x] += 1

    def on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h


def observation(b: Board) -> np.ndarray:
    """Seat 0's egocentric observation ``f32[2H-1, 2W-1, 13]`` (channels as in slinky)."""
    h, w = b.h, b.w
    # float32 arithmetic, and division as XLA compiles it (times the reciprocal),
    # so the observation matches the JAX encoder's bit for bit.
    per_cell = np.float32(1.0) / np.float32(h * w)
    health = b.health.astype(np.float32) * (np.float32(1.0) / np.float32(100.0))
    head_map = np.zeros((len(b.head), h, w), bool)
    for i, (x, y) in enumerate(b.head):
        if b.on_board(x, y):
            head_map[i, y, x] = True
    body = b.body.astype(np.float32)
    opp_body, opp_head = body[1:], head_map[1:]
    longer = (b.length[1:] >= b.length[0])[:, None, None]
    opp_health = np.where(opp_head, health[1:, None, None], np.float32(0.0))
    on = np.ones((h, w), np.float32)
    zeros = np.zeros((h, w), np.float32)
    obs = np.stack(
        [
            on,
            b.food.astype(np.float32),
            b.hazard,
            head_map[0].astype(np.float32),
            (body[0] > 0).astype(np.float32),
            body[0] * per_cell,
            (opp_head & longer).any(axis=0).astype(np.float32) if len(opp_body) else zeros,
            (opp_head & ~longer).any(axis=0).astype(np.float32) if len(opp_body) else zeros,
            (opp_body > 0).any(axis=0).astype(np.float32) if len(opp_body) else zeros,
            opp_body.max(axis=0) * per_cell if len(opp_body) else zeros,
            opp_health.max(axis=0) if len(opp_body) else zeros,
            on * health[0],
            on * (np.float32(b.length[0]) * per_cell),
        ],
        axis=-1,
    ).astype(np.float32)
    padded = np.pad(obs, ((h - 1, h - 1), (w - 1, w - 1), (0, 0)))
    hx, hy = int(np.clip(b.head[0, 0], 0, w - 1)), int(np.clip(b.head[0, 1], 0, h - 1))
    return padded[hy : hy + 2 * h - 1, hx : hx + 2 * w - 1]


def legal_moves(b: Board) -> np.ndarray:
    """Seat 0's ``bool[4]`` moves that don't certainly die to a wall or a body next turn."""

    def starving(i: int) -> bool:
        if b.health[i] > 1:
            return False
        x, y = b.head[i]
        return not any(b.on_board(x + dx, y + dy) and b.food[y + dy, x + dx] for dx, dy in DELTAS)

    solid = [i for i in range(len(b.head)) if not starving(i)]
    blocked = (b.body[solid] > 1).any(axis=0) if solid else np.zeros((b.h, b.w), bool)
    x, y = b.head[0]
    ok = np.array([b.on_board(x + dx, y + dy) and not blocked[y + dy, x + dx] for dx, dy in DELTAS])
    if starving(0) or not ok.any():
        return np.ones(4, bool)
    return ok


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """ReLU of a 3x3 "SAME" convolution (XLA's padding), ``x: [H, W, Cin]``."""
    h, wd, c = x.shape
    ho, wo = -(-h // stride), -(-wd // stride)
    ph, pw = max((ho - 1) * stride + 3 - h, 0), max((wo - 1) * stride + 3 - wd, 0)
    x = np.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    cols = np.concatenate(
        [
            x[dy : dy + (ho - 1) * stride + 1 : stride, dx : dx + (wo - 1) * stride + 1 : stride]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=-1,
    )
    return np.maximum(cols @ w.reshape(9 * c, -1) + b, 0.0)


class Net:
    """A trained baseline's network, from its checkpoint's ``params.npz`` and ``config.json``.

    What it scores the moves by depends on the checkpoint: policy logits (a
    ``policy`` head: PPO, REINFORCE, AlphaZero), Q-values (the DQN, plain or
    dueling) or the expected return of Rainbow's distributional head.
    """

    def __init__(self, params: str, config: dict[str, Any]) -> None:
        self.strides = tuple(config["conv_strides"])
        with np.load(params) as z:
            # The noisy layers' scales (``*_sigma``) are not needed to play.
            self.p = {k: z[k].astype(np.float32) for k in z.files if not k.endswith("_sigma")}
        if "policy/w" in self.p:
            self.kind = "policy"
        elif config.get("algorithm") == "rainbow":
            self.kind = "rainbow"
            self.support = np.linspace(
                config["v_min"], config["v_max"], config["num_atoms"], dtype=np.float32
            )
            self.dueling = bool(config["dueling"])
        else:
            self.kind = "q"
            self.dueling = bool(config.get("dueling", False))

    def _dense(self, name: str, x: np.ndarray) -> np.ndarray:
        return x @ self.p[f"{name}/w"] + self.p[f"{name}/b"]

    def scores(self, obs: np.ndarray) -> np.ndarray:
        """``f32[4]``: what the greedy move maximizes."""
        x = obs
        for i, stride in enumerate(self.strides):
            x = _conv3x3(x, self.p[f"conv{i}/w"], self.p[f"conv{i}/b"], stride)
        x = np.maximum(self._dense("hidden", x.reshape(-1)), 0.0)
        if self.kind == "policy":
            return self._dense("policy", x)
        if self.kind == "q":
            if not self.dueling:
                return self._dense("q", x)
            a = self._dense("advantage", x)
            return self._dense("value", x) + a - a.mean()
        atoms = len(self.support)
        if self.dueling:
            a = self._dense("advantage", x).reshape(4, atoms)
            logits = self._dense("value", x)[None, :] + a - a.mean(axis=0, keepdims=True)
        else:
            logits = self._dense("logits", x).reshape(4, atoms)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True) * self.support).sum(axis=-1)

    def move(self, b: Board) -> int:
        """The greedy legal move (the first one on ties, like ``jnp.argmax``)."""
        return int(np.argmax(np.where(legal_moves(b), self.scores(observation(b)), -np.inf)))


def is_duel(req: dict[str, Any]) -> bool:
    """Whether the request is the game the network learned: an 11x11 standard duel."""
    game, board = req.get("game", {}), req["board"]
    return (
        game.get("ruleset", {}).get("name", "standard") == "standard"
        and game.get("map", "standard") == "standard"
        and board["width"] == SIZE
        and board["height"] == SIZE
        and len(board["snakes"]) == 2
    )


def safe_move(req: dict[str, Any]) -> int:
    """A move that doesn't leave the board or enter a body segment that stays put.

    For games the network doesn't play; prefers moves away from the possible
    next heads of snakes at least as long.
    """
    board, you = req["board"], req["you"]
    w, h = board["width"], board["height"]
    blocked = {(p["x"], p["y"]) for s in board["snakes"] for p in s["body"][:-1]}
    me = next(s for s in board["snakes"] if s["id"] == you["id"])
    danger = {
        (s["body"][0]["x"] + dx, s["body"][0]["y"] + dy)
        for s in board["snakes"]
        if s["id"] != you["id"] and len(s["body"]) >= len(me["body"])
        for dx, dy in DELTAS
    }
    hx, hy = me["body"][0]["x"], me["body"][0]["y"]
    wrapped = "wrapped" in req.get("game", {}).get("ruleset", {}).get("name", "")
    best, best_score = 0, -1
    for a, (dx, dy) in enumerate(DELTAS):
        p = (hx + dx, hy + dy)
        if wrapped:
            p = (p[0] % w, p[1] % h)
        elif not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        score = 0 if p in blocked else 1 + (p not in danger)
        if score > best_score:
            best, best_score = a, score
    return best
