"""The PufferLib policy for the Battlesnake environment: ``baselines/ppo.py``'s network in PyTorch.

:class:`BattlesnakePolicy` is the same actor-critic as the JAX PPO baseline (3x3 "SAME"
convolutions, a dense layer, linear policy and value heads), so a network trained with
PufferLib's trainer (CPU, CUDA, or ROCm builds of PyTorch alike) can be written out with
:func:`export_ppo_checkpoint` and played by slinky's ``ppo:<run dir>`` agent against every
other agent in the repository.

* **Input.** One flat ``uint8`` vector per agent (``puffer/obs_format.py``): the egocentric
  board as raw integers, then the 4-byte move mask. The board is multiplied by
  ``channel_scale`` to get slinky's float observation.
* **Masking.** Illegal moves get the logit :data:`MASKED_LOGIT` inside the policy, so
  ``pufferlib.pytorch.sample_logits`` samples legal moves only and the log-probabilities
  and entropy the trainer sees are those of the masked policy, with no trainer change.
* **Layout.** Weights are stored in torch's layout and converted at export. The feature
  map is flattened in ``(H, W, C)`` order, as in ppo.py, so the dense weight needs no
  reordering; only the transposes below are applied.

Exported checkpoints keep ppo.py's file names and keys (``params.npz``: ``conv0/w`` as
``[3, 3, cin, cout]`` ... ``value/b``; ``config.json``). slinky's ppo agent builds its
network from ``PPOConfig``, whose game is an 11x11 board that is not wrapped, and checks
only the parameter shapes, which many other boards share (12x12, wrapped 21x21 ...) with
different channel scales: it would play such a network on the wrong board. So only a policy
for that board (any ruleset and snake count on it) is written as ``params.npz``; another
board's arrays go to ``puffer-params.npz``, which ``ppo:<run dir>`` does not read, and
:func:`load_ppo_checkpoint` reads it after checking the board recorded in ``config.json``
(``puffer_board``) against the environment's.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from puffer.obs_format import CHANNEL_NAMES, MASK_SIZE, NUM_CHANNELS, board_shape, channel_scale

# Logit of an illegal move, as ppo.py's MASKED_LOGIT: exp() underflows to exactly 0, but
# unlike -inf it keeps p * log p and every gradient finite.
MASKED_LOGIT = -1e9
NUM_ACTIONS = MASK_SIZE
# ppo.py's PPOConfig defaults, so a policy built without arguments exports as a default run.
DEFAULT_CONV_CHANNELS = (16, 32, 32)
DEFAULT_CONV_STRIDES = (2, 2, 1)
DEFAULT_HIDDEN = 256
DEFAULT_MAX_TURNS = 1000
KERNEL = 3
# The arrays of a policy for the board slinky's ppo agent plays, and of any other board.
PPO_PARAMS = "params.npz"
PUFFER_PARAMS = "puffer-params.npz"


def same_padding(size: int, stride: int) -> tuple[int, int]:
    """``(before, after)`` padding of XLA/TensorFlow "SAME" for a 3-wide kernel.

    The output has ``ceil(size / stride)`` cells and the padding is split with the extra
    cell after. torch's symmetric ``padding=1`` differs from this for stride 2 on an even
    size (it pads one cell before instead of after, which shifts the whole window grid),
    and ppo.py's weights were trained with "SAME".
    """
    out = -(-size // stride)
    total = max((out - 1) * stride + KERNEL - size, 0)
    return total // 2, total - total // 2


class SameConv2d(nn.Conv2d):
    """A 3x3 convolution with "SAME" padding (:func:`same_padding`) for a fixed input size.

    Symmetric padding goes to the convolution itself; an odd total (stride 2 on an even
    size) is asymmetric, and is padded explicitly first.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, in_size: tuple[int, int]
    ) -> None:
        pads = [same_padding(n, stride) for n in in_size]
        symmetric = all(before == after for before, after in pads)
        super().__init__(
            in_channels,
            out_channels,
            KERNEL,
            stride,
            padding=(pads[0][0], pads[1][0]) if symmetric else 0,
        )
        # F.pad takes the last dimension first: (left, right, top, bottom).
        self._pad = None if symmetric else (pads[1][0], pads[1][1], pads[0][0], pads[0][1])
        self.out_size = tuple(-(-n // stride) for n in in_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad is not None:
            x = F.pad(x, self._pad)
        return super().forward(x)


def _ints(values: Sequence[int], name: str, minimum: int = 1) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < minimum for v in out):
        raise ValueError(f"{name} must all be >= {minimum}, got {tuple(values)}")
    return out


class BattlesnakePolicy(nn.Module):
    """ppo.py's actor-critic on PufferLib's uint8 observations.

    Args:
      env: the :class:`~puffer.battlesnake.Battlesnake` environment, or anything with
        ``board_shape`` (``(rows, cols, 13)``), ``obs_size`` (``rows * cols * 13 + 4``)
        and ``channel_scale`` (``float32[13]``).
      conv_channels: output channels of the 3x3 convolutions.
      conv_strides: their strides (same length).
      hidden: width of the dense layer.

    ``forward_eval(obs)`` / ``forward(obs)`` map ``uint8[B, obs_size]`` to
    ``(logits float32[B, 4], value float32[B, 1])`` with the logits of illegal moves
    at :data:`MASKED_LOGIT`.
    """

    def __init__(
        self,
        env: Any,
        conv_channels: Sequence[int] = DEFAULT_CONV_CHANNELS,
        conv_strides: Sequence[int] = DEFAULT_CONV_STRIDES,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        rows, cols, channels = (int(n) for n in env.board_shape)
        if channels != NUM_CHANNELS:
            raise ValueError(
                f"board_shape {tuple(env.board_shape)} must have {NUM_CHANNELS} channels"
            )
        self.board_size = rows * cols * channels
        if int(env.obs_size) != self.board_size + MASK_SIZE:
            raise ValueError(
                f"obs_size {env.obs_size} != {rows}x{cols}x{channels} board + {MASK_SIZE} mask"
            )
        scale = np.asarray(env.channel_scale, np.float32)
        if scale.shape != (NUM_CHANNELS,):
            raise ValueError(f"channel_scale must have shape ({NUM_CHANNELS},), got {scale.shape}")
        conv_channels = _ints(conv_channels, "conv_channels")
        conv_strides = _ints(conv_strides, "conv_strides")
        if len(conv_channels) != len(conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        hidden = _ints([hidden], "hidden")[0]

        self.board_shape = (rows, cols, channels)
        self.obs_size = self.board_size + MASK_SIZE
        self.conv_channels = conv_channels
        self.conv_strides = conv_strides
        self.hidden_size = hidden
        # Not in the state dict: it follows from the environment, not from training.
        self.register_buffer("channel_scale", torch.from_numpy(scale.copy()), persistent=False)

        convs = []
        size, cin = (rows, cols), channels
        for cout, stride in zip(conv_channels, conv_strides, strict=True):
            conv = SameConv2d(cin, cout, stride, size)
            convs.append(conv)
            size, cin = conv.out_size, cout
        self.convs = nn.ModuleList(convs)
        self.dense = nn.Linear(size[0] * size[1] * cin, hidden)
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)
        self.init_weights()

    def init_weights(self) -> None:
        """ppo.py's ``init_network``: He-normal torso, zero biases, a near-zero policy head.

        The policy head's std is ``sqrt(1e-4 / hidden)`` so play starts uniform over the
        legal moves; the value head's is ``sqrt(1 / hidden)``.
        """
        with torch.no_grad():
            for conv in self.convs:
                nn.init.normal_(
                    conv.weight, 0.0, math.sqrt(2.0 / (KERNEL * KERNEL * conv.in_channels))
                )
            nn.init.normal_(self.dense.weight, 0.0, math.sqrt(2.0 / self.dense.in_features))
            nn.init.normal_(self.policy_head.weight, 0.0, math.sqrt(1e-4 / self.hidden_size))
            nn.init.normal_(self.value_head.weight, 0.0, math.sqrt(1.0 / self.hidden_size))
            for layer in (*self.convs, self.dense, self.policy_head, self.value_head):
                nn.init.zeros_(layer.bias)

    def forward_eval(
        self, observations: torch.Tensor, state: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(masked logits [B, 4], value [B, 1])`` for ``uint8[B, obs_size]`` observations.

        ``state`` is PufferLib's recurrent-state dict, unused by this feed-forward net.
        """
        batch = observations.shape[0]
        observations = observations.reshape(batch, -1)
        if observations.shape[1] != self.obs_size:
            raise ValueError(
                f"expected observations of size {self.obs_size}, got {tuple(observations.shape)}"
            )
        rows, cols, channels = self.board_shape
        board = observations[:, : self.board_size].reshape(batch, rows, cols, channels)
        mask = observations[:, self.board_size :]
        x = (board.to(torch.float32) * self.channel_scale).permute(0, 3, 1, 2)  # NHWC -> NCHW
        for conv in self.convs:
            x = F.relu(conv(x))
        x = F.relu(self.dense(x.permute(0, 2, 3, 1).reshape(batch, -1)))  # flatten as (H, W, C)
        # float() first: under autocast the heads are half precision, which can't hold -1e9.
        logits = self.policy_head(x).float().masked_fill(mask == 0, MASKED_LOGIT)
        return logits, self.value_head(x).float()

    def forward(
        self, observations: torch.Tensor, state: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(observations, state)


# --- Checkpoints in baselines/ppo.py's format -------------------------------------------


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())


def ppo_params(policy: BattlesnakePolicy) -> dict[str, np.ndarray]:
    """The parameters as ppo.py's flattened ``layer/name`` arrays (``float32``).

    A torch convolution weight is ``[cout, cin, kh, kw]`` and ppo.py's is
    ``[kh, kw, cin, cout]``; a torch linear weight is ``[out, in]`` and ppo.py's
    ``[in, out]``.
    """
    params = {}
    for i, conv in enumerate(policy.convs):
        params[f"conv{i}/w"] = _to_numpy(conv.weight.permute(2, 3, 1, 0))
        params[f"conv{i}/b"] = _to_numpy(conv.bias)
    for name, layer in (
        ("hidden", policy.dense),
        ("policy", policy.policy_head),
        ("value", policy.value_head),
    ):
        params[f"{name}/w"] = _to_numpy(layer.weight.T)
        params[f"{name}/b"] = _to_numpy(layer.bias)
    return params


def set_ppo_params(policy: BattlesnakePolicy, params: Mapping[str, np.ndarray]) -> None:
    """Load :func:`ppo_params`-style arrays into ``policy`` (the inverse).

    Raises:
      ValueError: a key is missing or unexpected, or an array has the wrong shape for
        this policy's network (e.g. a checkpoint of another board size or architecture).
    """
    expected = ppo_params(policy)
    if missing := sorted(set(expected) - set(params)):
        raise ValueError(f"checkpoint is missing parameters {missing}")
    if extra := sorted(set(params) - set(expected)):
        raise ValueError(f"checkpoint has unexpected parameters {extra}")
    arrays = {k: np.asarray(params[k], np.float32) for k in expected}
    for key, ref in expected.items():
        if arrays[key].shape != ref.shape:
            raise ValueError(f"{key}: checkpoint shape {arrays[key].shape} != {ref.shape}")

    def tensor(key: str, like: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(arrays[key]).to(like.device, like.dtype)

    with torch.no_grad():
        for i, conv in enumerate(policy.convs):
            conv.weight.copy_(tensor(f"conv{i}/w", conv.weight).permute(3, 2, 0, 1))
            conv.bias.copy_(tensor(f"conv{i}/b", conv.bias))
        for name, layer in (
            ("hidden", policy.dense),
            ("policy", policy.policy_head),
            ("value", policy.value_head),
        ):
            layer.weight.copy_(tensor(f"{name}/w", layer.weight).T)
            layer.bias.copy_(tensor(f"{name}/b", layer.bias))


def _json_default(value: Any) -> Any:
    """Numpy scalars and arrays (e.g. from an env's ``game_config``) as plain JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def board_record(shape: Sequence[int], scale: Sequence[float]) -> dict[str, Any]:
    """A board as ``config.json`` records it: the observation shape and the channel scale.

    Together they tell boards apart that the network's parameter shapes do not (the scale
    holds the cell count, which wrapped 21x21 and 11x11 boards do not share).
    """
    return {"shape": [int(n) for n in shape], "channel_scale": [float(x) for x in scale]}


def ppo_board() -> dict[str, Any]:
    """The board of slinky's ppo agent (``PPOConfig``'s game): 11x11, not wrapped."""
    return board_record(board_shape(11, 11), channel_scale(11, 11))


def policy_board(policy: BattlesnakePolicy) -> dict[str, Any]:
    return board_record(policy.board_shape, policy.channel_scale.cpu().numpy())


def _describe_board(board: Mapping[str, Any]) -> str:
    cells = round(1 / board["channel_scale"][CHANNEL_NAMES.index("own_countdown")])
    return f"{tuple(board['shape'])} observations on {cells} cells"


def _write_atomically(path: str, write: Any) -> None:
    """``write(tmp_path)``, then rename over ``path``: readers never see half a file."""
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    write(tmp)
    os.replace(tmp, path)


def export_ppo_checkpoint(
    policy: BattlesnakePolicy, run_dir: str | os.PathLike, **config: Any
) -> str:
    """Write ``run_dir/params.npz`` and ``run_dir/config.json`` in baselines/ppo.py's format.

    The result is a run directory for slinky's ``ppo:<run dir>`` agent, if the policy is for
    the 11x11 board that agent plays. For any other board the arrays are written to
    ``puffer-params.npz`` instead (see the module docstring) and ``ppo:<run dir>`` finds no
    checkpoint. ``config.json`` holds the network (``conv_channels``, ``conv_strides``,
    ``hidden``), ``max_turns`` (default 1000, ppo.py's: the turn cap slinky's evaluation
    applies to the agent; ``None``, which ``Battlesnake.game_config`` spells "no limit", is
    the default too), the policy's board (``puffer_board``) and every keyword in ``config``
    as given. ``baselines/ppo.py: load_config`` reads the keys it knows (any other, such as
    PufferLib trainer settings, it ignores), so a keyword named like a ``PPOConfig`` field
    (``seed``, ``gamma``, ``num_envs`` ...) is read as that field. Each file is written
    atomically, and the config is serialized first: a value JSON cannot hold raises before
    any file is touched.

    Returns:
      The path of the parameter file.

    Raises:
      ValueError: ``config`` contradicts the policy's network, or ``max_turns`` < 1
        (``PPOConfig`` refuses it; a game without a cap is evaluated with a cap anyway).
      TypeError: a ``config`` value is not JSON serializable.
    """
    network = {
        "conv_channels": list(policy.conv_channels),
        "conv_strides": list(policy.conv_strides),
        "hidden": policy.hidden_size,
    }
    for key, value in network.items():
        if key in config and json.loads(json.dumps(config[key], default=_json_default)) != value:
            raise ValueError(f"config {key}={config[key]!r} contradicts the policy ({value!r})")
    board = policy_board(policy)
    record = {**network, "max_turns": DEFAULT_MAX_TURNS, **config, "puffer_board": board}
    if record["max_turns"] is None:
        record["max_turns"] = DEFAULT_MAX_TURNS
    if int(record["max_turns"]) < 1:
        raise ValueError(f"max_turns must be >= 1 for ppo.py's config, got {record['max_turns']!r}")
    text = json.dumps(record, indent=2, default=_json_default)

    playable = board == ppo_board()
    os.makedirs(run_dir, exist_ok=True)
    params_path = os.path.join(run_dir, PPO_PARAMS if playable else PUFFER_PARAMS)
    _write_atomically(params_path, lambda tmp: np.savez(tmp, **ppo_params(policy)))

    def write_config(tmp: str) -> None:
        with open(tmp, "w") as f:
            f.write(text)

    _write_atomically(os.path.join(run_dir, "config.json"), write_config)
    # An earlier export of the other kind would be read by ppo:<run dir> with this config.
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(run_dir, PUFFER_PARAMS if playable else PPO_PARAMS))
    return params_path


def load_ppo_checkpoint(run_dir: str | os.PathLike, env: Any) -> BattlesnakePolicy:
    """The policy saved in ``run_dir`` (by :func:`export_ppo_checkpoint` or ppo.py) for ``env``.

    The network comes from ``config.json`` (ppo.py's defaults for missing keys) and the
    board from ``env``. The board the checkpoint was exported for (``puffer_board``; a ppo.py
    run, which has none, is the 11x11 board) must be ``env``'s: parameter shapes alone do not
    tell boards apart.

    Raises:
      ValueError: the checkpoint was exported for another board than ``env``'s, or its
        parameters do not fit the network (see :func:`set_ppo_params`).
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    policy = BattlesnakePolicy(
        env,
        conv_channels=config.get("conv_channels", DEFAULT_CONV_CHANNELS),
        conv_strides=config.get("conv_strides", DEFAULT_CONV_STRIDES),
        hidden=config.get("hidden", DEFAULT_HIDDEN),
    )
    board, saved = policy_board(policy), config.get("puffer_board", ppo_board())
    if saved != board:
        raise ValueError(
            f"{run_dir} was exported for a board of {_describe_board(saved)}, "
            f"but the environment's has {_describe_board(board)}"
        )
    params = PPO_PARAMS if board == ppo_board() else PUFFER_PARAMS
    with np.load(os.path.join(run_dir, params)) as f:
        set_ppo_params(policy, {key: f[key] for key in f.files})
    return policy
