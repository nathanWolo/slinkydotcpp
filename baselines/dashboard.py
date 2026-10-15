"""A live training dashboard for the baselines (DQN, PPO, REINFORCE, Rainbow, AlphaZero).

Including the multi-mode runs of ``ppo_modes``, ``reinforce_modes``, ``dqn_modes``,
``rainbow_modes`` and ``az_modes`` (``config.json`` has ``"algorithm": "<name>_modes"``): each
is charted like its single-mode namesake, plus a chart per game mode (``game_modes`` blocks of
the train records, and the evals' ``game_mode``). Their step counts are env steps *per mode*.

::

    python baselines/dashboard.py                    # then open http://127.0.0.1:8050
    python baselines/dashboard.py --port 9000 runs/ other/runs/

It serves ``dashboard.html`` and the ``metrics.jsonl`` / ``config.json`` of every
run directory it finds (any directory with a ``metrics.jsonl``, searched up to
two levels below each root; default roots: ``runs/`` and
``baselines/checkpoints/``). The page re-reads them every few seconds, so a run
in progress updates live.

Links to a run: ``http://host:8050/#run=<run>`` opens the Training tab with that run
(and only that run) plotted, where ``<run>`` is the run's id (the ``id`` of ``/api/runs``:
its directory, relative to the repository if inside it), or just the name of its directory
(``#run=dqn-modes-M-s0``; the most recently updated if several roots have one of that name).
The page keeps the hash up to date while exactly one run is plotted. ``#games`` and
``#games=<replay id>`` open the Games tab.

A multi-mode run's ``total_env_steps`` is a cap while it is in its main phase (the plateau
watcher of ``benchmarks/plateau.py`` may stop it sooner) and the run's end once ``cooldown_from``
is set; ``/api/runs`` says which in ``phase``.

Its Games tab plays recorded games: every ``slinky-replay/1`` JSON file (as
written by ``python -m slinky.replay --out``) up to two levels below the replay
roots (default ``replays/``; ``--replays`` to change), shown in the replay
viewer (``src/slinky/viewer.html``).

Standard library only; it binds to localhost unless ``--host`` says otherwise,
answers only requests addressed to that host (so a web page can't reach it
through DNS rebinding), and serves nothing but the page, those files and the
viewer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("dashboard.html")
DEFAULT_ROOTS = (REPO_ROOT / "runs", REPO_ROOT / "baselines" / "checkpoints")
DEFAULT_REPLAY_ROOTS = (REPO_ROOT / "replays",)
VIEWER = REPO_ROOT / "src" / "slinky" / "viewer.html"
REPLAY_FORMAT = "slinky-replay/1"
# The viewer's data slot, as slinky.replay.render_html fills it.
_DATA_TAG = '<script type="application/json" id="replay-data">null</script>'
# A run whose metrics changed this recently counts as live (or within 3 of its own
# log intervals, for slow runs).
LIVE_SECONDS = 180.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def find_runs(roots: list[Path]) -> dict[str, Path]:
    """``{run id: directory}`` for directories with a ``metrics.jsonl`` (ids are display paths)."""
    runs: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        found = [*root.glob("metrics.jsonl"), *root.glob("*/metrics.jsonl")]
        for metrics in sorted([*found, *root.glob("*/*/metrics.jsonl")]):
            run = metrics.parent.resolve()
            runs.setdefault(_display(run), run)
    return runs


def _display(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def read_config(run: Path) -> dict[str, Any]:
    try:
        config = json.loads((run / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def read_records(run: Path) -> list[dict[str, Any]]:
    """The run's metric records; a half-written last line (a run mid-write) is skipped."""
    try:
        lines = (run / "metrics.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def read_simsweep(run: Path) -> list[dict[str, Any]]:
    """The run's strength-vs-search points (``simsweep.jsonl``): the last line per point wins.

    Parsed as tolerantly as :func:`read_records`; a missing file gives ``[]``.
    """
    try:
        lines = (run / "simsweep.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return []
    points: dict[Any, dict[str, Any]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("type") != "simsweep":
            continue
        key = record.get("point")
        if key is None:
            fields = ("label", "env_steps", "game_mode", "simulations", "play")
            key = tuple(str(record.get(k)) for k in fields)
        points[key] = record  # re-assigning a key keeps its first-seen position
    return list(points.values())


def algorithm(config: dict[str, Any]) -> str:
    # dqn.py's and ppo.py's config.json have no "algorithm" key (rainbow.py's,
    # reinforce.py's, az.py's and the multi-mode trainers' do: "ppo_modes", "reinforce_modes",
    # "dqn_modes", "rainbow_modes", "az_modes").
    if not config:
        return "unknown"
    return str(config.get("algorithm") or ("ppo" if "gae_lambda" in config else "dqn"))


def is_modes(config: dict[str, Any]) -> bool:
    """A multi-mode run (``<algorithm>_modes``): it trains on every game mode at once."""
    return str(config.get("algorithm")).endswith("_modes")


# Algorithms whose update is a batch of whole games, so a chunk is as long as the games in it.
WHOLE_GAME_ALGORITHMS = ("reinforce", "reinforce_modes")


def total_steps(config: dict[str, Any]) -> int | None:
    """The env steps a run will take: ``total_env_steps`` rounded up to whole log chunks.

    A chunk is whole env-step batches (``num_envs`` each; ``num_envs * num_steps``
    for PPO's and AlphaZero's iterations), as many as fit in ``log_every`` and at least
    one. The multi-mode DQN, Rainbow and AlphaZero count env steps per mode and chunk the
    same way. REINFORCE's chunks (also ``reinforce_modes``') are whole batches of games, of
    varying length: its runs stop at the first chunk that reaches ``total_env_steps``.

    A multi-mode run's cooldown (``cooldown_from`` >= 0, ``total_env_steps`` = that + the
    cooldown) continues the chunks of its main phase, which stopped at a chunk boundary, so
    the same rounding holds.
    """
    try:
        total = int(config["total_env_steps"])
    except (KeyError, TypeError, ValueError):
        return None
    if config.get("algorithm") in WHOLE_GAME_ALGORITHMS:
        return total
    try:
        every, envs = (int(config[k]) for k in ("log_every", "num_envs"))
        step = max(envs, 1) * max(int(config.get("num_steps", 1)), 1)
    except (KeyError, TypeError, ValueError):
        return None
    chunk = max(every // step, 1) * step
    return -(-total // chunk) * chunk


def phase(config: dict[str, Any]) -> str | None:
    """Where a multi-mode run is: ``"main"``, ``"cooldown"`` or None (not such a run).

    In the main phase the learning rate is constant and the plateau watcher may stop the run
    before ``total_env_steps``, which is then only a cap. From ``cooldown_from`` on the
    learning rate decays to 0 at ``total_env_steps``, which is then where the run ends. A run
    that anneals its learning rate over ``total_env_steps`` (``ppo_modes --anneal-lr``) has no
    phases: it runs to its end.
    """
    if not is_modes(config):
        return None
    try:
        if int(config.get("cooldown_from", -1)) >= 0:
            return "cooldown"
    except (TypeError, ValueError):
        pass
    return None if config.get("anneal_lr") else "main"


def is_headline_eval(record: dict[str, Any], config: dict[str, Any] | None = None) -> bool:
    """An eval of the main agent against ``random_legal``, which every baseline logs.

    The main agent is the greedy policy, or for AlphaZero the network with search
    (mode "search"). PPO and REINFORCE also evaluate against other opponents and with
    a sampled policy, AlphaZero also the raw network; DQN and Rainbow records have
    neither field.

    A ``ppo_modes`` run evaluates every game mode and tags its records with ``game_mode``
    (their ``score`` is the share of the win). Its headline is the first of the run's
    ``game_modes`` (from ``config``), the mode its top-level train metrics follow too; the
    other modes have their own charts. Without a config that names the run's modes, a
    record with a ``game_mode`` is not a headline.
    """
    if record.get("opponent", "random_legal") != "random_legal" or record.get(
        "mode", "greedy"
    ) not in ("greedy", "search"):
        return False
    game_mode = record.get("game_mode")
    return game_mode is None or game_mode == headline_game_mode(config)


def headline_game_mode(config: dict[str, Any] | None) -> str | None:
    """The first of a multi-mode run's ``game_modes`` (None: not a multi-mode run's config)."""
    modes = (config or {}).get("game_modes")
    return modes[0] if isinstance(modes, list) and modes and isinstance(modes[0], str) else None


def read_run(run: Path) -> tuple[list[dict[str, Any]], float, str]:
    """``(records, mtime, version)`` of the run's metrics.

    The file is stat'ed *before* it is read: a write that lands in between makes the
    version older than the records, which costs the page one extra fetch, never a
    missed update. The version includes the size, which changes with every append.
    """
    mtime, version = _stat_metrics(run)
    return read_records(run), mtime, version


def _stat_metrics(run: Path) -> tuple[float, str]:
    """``(mtime, version)`` of the run's ``metrics.jsonl`` (see :func:`read_run`)."""
    try:
        st = (run / "metrics.jsonl").stat()
    except OSError:
        return 0.0, "missing"
    version = f"{st.st_mtime_ns}:{st.st_size}"
    try:  # a sweep job writes simsweep.jsonl: it changes the version, not the mtime ("live")
        sw = (run / "simsweep.jsonl").stat()
        version += f"|sw{sw.st_mtime_ns}:{sw.st_size}"
    except OSError:
        pass
    return st.st_mtime, version


def _number(x: Any) -> int | float | None:
    return x if isinstance(x, int | float) and not isinstance(x, bool) else None


def _digest(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """What :func:`run_summary` takes from a run's records."""
    train = [r for r in records if r.get("type") == "train"]
    evals = [r for r in records if r.get("type") == "eval"]
    headline = [r for r in evals if is_headline_eval(r, config)]
    return {
        "steps": (_number(train[-1].get("env_steps")) or 0) if train else 0,
        # The last chunk's and eval's wall time: how often this run writes.
        "cadence": sum(_number(r.get("seconds")) or 0.0 for r in (train[-1:] + evals[-1:])),
        # The chunks' wall time, summed: unlike "elapsed" it survives a resume or a reboot.
        "train_seconds": sum(_number(r.get("seconds")) or 0.0 for r in train),
        "last_train": train[-1] if train else None,
        "last_eval": headline[-1] if headline else None,
    }


# Run directory -> (metrics version and headline game mode, digest). The run list reads every
# run on every poll (and a campaign has dozens, of megabytes each), but most runs are
# finished: their digest is kept until their metrics.jsonl changes.
_digests: dict[Path, tuple[tuple[str, str | None], dict[str, Any]]] = {}


def run_summary(run_id: str, run: Path, read: tuple | None = None) -> dict[str, Any]:
    """What the run list shows: identity, status, progress and the latest numbers."""
    config = read_config(run)
    if read is None:
        mtime, version = _stat_metrics(run)  # before the read, as in read_run
        key = (version, headline_game_mode(config))
        cached = _digests.get(run)
        if cached is None or cached[0] != key:
            cached = _digests[run] = (key, _digest(read_records(run), config))
        digest = cached[1]
    else:
        records, mtime, version = read
        digest = _digest(records, config)
    total = total_steps(config)
    steps = digest["steps"]
    # DONE: the campaign's marker of a run that is completely finished (its cooldown included).
    finished = (total is not None and steps >= total) or (run / "DONE").exists()
    return {
        "id": run_id,
        "name": run.name,
        "algorithm": algorithm(config),
        # "main" (total_steps is a cap), "cooldown" (it is the end) or None: see phase().
        "phase": phase(config),
        "updated": mtime,
        "version": version,
        # Live: written to within LIVE_SECONDS, or within 3 of this run's own chunk + eval
        # times (a slow run logs less often).
        "live": time.time() - mtime < max(LIVE_SECONDS, 3 * digest["cadence"]),
        "finished": finished,
        "env_steps": steps,
        "total_steps": total,
        "train_seconds": digest["train_seconds"],
        "last_train": digest["last_train"],
        "last_eval": digest["last_eval"],
    }


def find_replays(roots: list[Path]) -> dict[str, Path]:
    """``{replay id: file}`` for the ``*.json`` files up to two levels below each root.

    Which of them are replays is decided by :func:`read_replay`; a root may also be
    one file.
    """
    found: dict[str, Path] = {}
    for root in roots:
        if root.is_file():
            files = [root] if root.suffix == ".json" else []
        elif root.is_dir():
            files = [*root.glob("*.json"), *root.glob("*/*.json"), *root.glob("*/*/*.json")]
        else:
            continue
        for path in sorted(files):
            found.setdefault(_display(path.resolve()), path.resolve())
    return found


_replay_cache: dict[Path, tuple[str, dict[str, Any] | None]] = {}


def read_replay(path: Path) -> dict[str, Any] | None:
    """The replay in ``path``, or None if it isn't one (cached until the file changes)."""
    try:
        st = path.stat()
    except OSError:
        return None
    stamp = f"{st.st_mtime_ns}:{st.st_size}"
    cached = _replay_cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        replay = None
    if not (
        isinstance(replay, dict)
        and replay.get("format") == REPLAY_FORMAT
        and isinstance(replay.get("games"), list)
        and isinstance(replay.get("agents"), list)
    ):
        replay = None
    _replay_cache[path] = (stamp, replay)
    return replay


def replay_summary(replay_id: str, path: Path, replay: dict[str, Any]) -> dict[str, Any]:
    """What the Games tab lists for a replay file: who played and how it went."""
    agents = [str(a) for a in replay["agents"]]
    wins = dict.fromkeys(agents, 0)
    draws = cut = turns = 0
    for game in replay["games"]:
        result, seats = game.get("result") or {}, game.get("seats") or []
        winner = result.get("winner")
        if isinstance(winner, int) and 0 <= winner < len(seats):
            name = str(seats[winner])
            wins[name] = wins.get(name, 0) + 1
        draws += bool(result.get("draw"))
        cut += bool(result.get("truncated"))
        turns += _number(result.get("turns")) or 0
    games = len(replay["games"])
    board = replay.get("board") or {}
    try:
        updated = path.stat().st_mtime
    except OSError:
        updated = 0.0
    return {
        "id": replay_id,
        "name": path.stem,
        "agents": agents,
        "games": games,
        "wins": wins,
        "draws": draws,
        "truncated": cut,
        "mean_turns": turns / games if games else None,
        "max_turns": replay.get("max_turns"),
        "board": f"{board.get('width', '?')}×{board.get('height', '?')}",
        "ruleset": replay.get("ruleset"),
        "updated": updated,
    }


def viewer_page(replay: dict[str, Any]) -> bytes:
    """The replay viewer with ``replay`` embedded, as ``slinky.replay.render_html`` makes it."""
    template = VIEWER.read_text(encoding="utf-8")
    if template.count(_DATA_TAG) != 1:
        raise RuntimeError("viewer.html must contain the replay-data placeholder exactly once")
    # "<" -> \u003c keeps "</script>" and "<!--" in names from closing the tag; still JSON.
    data = json.dumps(replay, separators=(",", ":")).replace("<", "\\u003c")
    fragment = template.replace(_DATA_TAG, _DATA_TAG.replace(">null<", f">{data}<"))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>\n{fragment}\n</body>\n</html>\n"
    ).encode()


class Handler(BaseHTTPRequestHandler):
    roots: list[Path] = []
    replay_roots: list[Path] = list(DEFAULT_REPLAY_ROOTS)
    hosts: set[str] | None = set(LOCAL_HOSTS)  # accepted Host headers; None accepts any

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        if not self._host_allowed():
            self._send(b"wrong host", "text/plain", HTTPStatus.MISDIRECTED_REQUEST)
            return
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            summaries = []
            for run_id, run in find_runs(self.roots).items():
                try:
                    summaries.append(run_summary(run_id, run))
                except Exception:  # one odd run must not hide the others
                    continue
            self._json(summaries)
        elif url.path == "/api/metrics":
            run_id = parse_qs(url.query).get("run", [""])[0]
            run = find_runs(self.roots).get(run_id)  # only runs we listed: no arbitrary paths
            if run is None:
                self._json({"error": f"unknown run {run_id!r}"}, HTTPStatus.NOT_FOUND)
                return
            read = read_run(run)
            self._json(
                {**run_summary(run_id, run, read), "config": read_config(run),
                    "records": read[0], "simsweep": read_simsweep(run)}
            )
        elif url.path == "/api/replays":
            listed = []
            for replay_id, path in find_replays(self.replay_roots).items():
                try:
                    replay = read_replay(path)
                    if replay is not None:
                        listed.append(replay_summary(replay_id, path, replay))
                except Exception:  # one odd file must not hide the others
                    continue
            listed.sort(key=lambda r: (-r["updated"], r["id"]))
            self._json(listed)
        elif url.path == "/viewer":
            replay_id = parse_qs(url.query).get("replay", [""])[0]
            path = find_replays(self.replay_roots).get(replay_id)  # only files we list
            replay = read_replay(path) if path is not None else None
            if replay is None:
                self._send(b"unknown replay", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                page = viewer_page(replay)
            except (OSError, RuntimeError) as e:
                self._send(
                    f"viewer unavailable: {e}".encode(),
                    "text/plain",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send(page, "text/html; charset=utf-8")
        else:
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def _host_allowed(self) -> bool:
        """Only requests addressed to this server: a DNS-rebinding page sends its own name."""
        if self.hosts is None:
            return True
        host = self.headers.get("Host", "")
        if host.startswith("["):  # [::1]:8050
            host = host[1 : host.find("]")] if "]" in host else host
        elif host.count(":") == 1:
            host = host.rsplit(":", 1)[0]
        return host.lower() in self.hosts

    def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        # NaN/inf (e.g. a diverged loss) are not JSON; send them as null.
        body = json.dumps(_finite(data), separators=(",", ":")).encode()
        self._send(body, "application/json", status)

    def _send(self, body: bytes, kind: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # browser went away
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # the page polls every few seconds; keep the terminal quiet


def _finite(x: Any) -> Any:
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve a live dashboard of baseline training runs.")
    p.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="directories to search for runs (default: runs/ and baselines/checkpoints/)",
    )
    p.add_argument(
        "--replays",
        action="append",
        type=Path,
        default=[],
        help="directory (or file) of replay JSON for the Games tab; repeatable (default: replays/)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="also answer requests addressed to this host name (e.g. the machine's name when "
        "serving on the network with --host 0.0.0.0); repeatable",
    )
    args = p.parse_args(argv)
    Handler.roots = [r.resolve() for r in args.roots] or list(DEFAULT_ROOTS)
    Handler.replay_roots = [r.resolve() for r in args.replays] or list(DEFAULT_REPLAY_ROOTS)
    Handler.hosts = {*LOCAL_HOSTS, args.host.lower(), *(h.lower() for h in args.allow_host)}
    if args.host in ("0.0.0.0", "::") and not args.allow_host:
        Handler.hosts = None  # serving on every interface: any name may be used
    family = socket.getaddrinfo(args.host, args.port, type=socket.SOCK_STREAM)[0][0]
    server_class = type("Server", (ThreadingHTTPServer,), {"address_family": family})
    server = server_class((args.host, args.port), Handler)
    found = find_runs(Handler.roots)
    shown = f"[{args.host}]" if ":" in args.host else args.host
    print(f"{len(found)} run(s) under {', '.join(_display(r) for r in Handler.roots)}")
    replays = find_replays(Handler.replay_roots)
    shown_roots = ", ".join(_display(r) for r in Handler.replay_roots)
    print(f"{len(replays)} JSON file(s) for the Games tab under {shown_roots}")
    print(f"dashboard: http://{shown}:{server.server_port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
