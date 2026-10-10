"""A live training dashboard for the baselines (DQN, Rainbow) in the browser.

::

    python baselines/dashboard.py                    # then open http://127.0.0.1:8050
    python baselines/dashboard.py --port 9000 runs/ other/runs/

It serves ``dashboard.html`` and the ``metrics.jsonl`` / ``config.json`` of every
run directory it finds (any directory with a ``metrics.jsonl``, searched up to
two levels below each root; default roots: ``runs/`` and
``baselines/checkpoints/``). The page re-reads them every few seconds, so a run
in progress updates live. Standard library only; it binds to localhost unless
``--host`` says otherwise, answers only requests addressed to that host (so a
web page can't reach it through DNS rebinding), and serves nothing but the page
and those two files.
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


def total_steps(config: dict[str, Any]) -> int | None:
    """The env steps a run will take: ``total_env_steps`` rounded up to whole log chunks."""
    try:
        total, every, envs = (int(config[k]) for k in ("total_env_steps", "log_every", "num_envs"))
    except (KeyError, TypeError, ValueError):
        return None
    chunk = max(every // max(envs, 1), 1) * max(envs, 1)
    return -(-total // chunk) * chunk


def read_run(run: Path) -> tuple[list[dict[str, Any]], float, str]:
    """``(records, mtime, version)`` of the run's metrics.

    The file is stat'ed *before* it is read: a write that lands in between makes the
    version older than the records, which costs the page one extra fetch, never a
    missed update. The version includes the size, which changes with every append.
    """
    try:
        st = (run / "metrics.jsonl").stat()
        mtime, version = st.st_mtime, f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        mtime, version = 0.0, "missing"
    return read_records(run), mtime, version


def _number(x: Any) -> int | float | None:
    return x if isinstance(x, int | float) and not isinstance(x, bool) else None


def run_summary(run_id: str, run: Path, read: tuple | None = None) -> dict[str, Any]:
    """What the run list shows: identity, status, progress and the latest numbers."""
    config = read_config(run)
    records, mtime, version = read_run(run) if read is None else read
    train = [r for r in records if r.get("type") == "train"]
    evals = [r for r in records if r.get("type") == "eval"]
    total = total_steps(config)
    steps = (_number(train[-1].get("env_steps")) or 0) if train else 0
    # Live: written to within LIVE_SECONDS, or within 3 of this run's own chunk + eval
    # times (a slow run logs less often).
    cadence = sum(_number(r.get("seconds")) or 0.0 for r in (train[-1:] + evals[-1:]))
    return {
        "id": run_id,
        "name": run.name,
        # dqn.py's config.json predates the "algorithm" key.
        "algorithm": config.get("algorithm", "dqn" if config else "unknown"),
        "updated": mtime,
        "version": version,
        "live": time.time() - mtime < max(LIVE_SECONDS, 3 * cadence),
        "finished": total is not None and steps >= total,
        "env_steps": steps,
        "total_steps": total,
        "last_train": train[-1] if train else None,
        "last_eval": evals[-1] if evals else None,
    }


class Handler(BaseHTTPRequestHandler):
    roots: list[Path] = []
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
                {**run_summary(run_id, run, read), "config": read_config(run), "records": read[0]}
            )
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
    Handler.hosts = {*LOCAL_HOSTS, args.host.lower(), *(h.lower() for h in args.allow_host)}
    if args.host in ("0.0.0.0", "::") and not args.allow_host:
        Handler.hosts = None  # serving on every interface: any name may be used
    family = socket.getaddrinfo(args.host, args.port, type=socket.SOCK_STREAM)[0][0]
    server_class = type("Server", (ThreadingHTTPServer,), {"address_family": family})
    server = server_class((args.host, args.port), Handler)
    found = find_runs(Handler.roots)
    shown = f"[{args.host}]" if ":" in args.host else args.host
    print(f"{len(found)} run(s) under {', '.join(_display(r) for r in Handler.roots)}")
    print(f"dashboard: http://{shown}:{server.server_port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
