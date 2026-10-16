# slinky-server-native

The Battlesnake HTTP server for the native az_modes engine. It is a drop-in replacement for
the Python ladder server (`python -m slinky.server`, `src/slinky/server.py`): the same routes,
the same response bodies byte for byte, the same `/start` cache, the same move log, the same
log lines and the same tie-noise seeds. A native server and a Python server that get the same
requests in the same order answer with the same moves (measured below).

Package `slinky-server-native`, binary `slinky-native-server`.

## Build

```bash
cd /home/nathan/slinky-native/native
source .cargo-env.sh
export CARGO_TARGET_DIR=/dev/shm/naz-target-server
cargo build --release -p slinky-server-native
# -> $CARGO_TARGET_DIR/release/slinky-native-server (about 1.6 MB, no runtime dependencies)
```

The workspace builds release binaries with `panic = "abort"`. The server never panics on a
request: the api crate's conversion and the search return errors, and every error becomes the
safe move. As a second line of defence, the search threads catch panics. That only works in an
unwinding build, so build the deployed binary with:

```bash
CARGO_PROFILE_RELEASE_PANIC=unwind cargo build --release -p slinky-server-native
```

If a search panics in that build, the request gets the safe move and the server keeps
running. In an abort build, the process exits and systemd restarts it (`Restart=always`, 5 s).

## Run

```text
slinky-native-server --snake NAME=RUN_DIR[:simulations=N][:budget_ms=T] [--snake ...]
    [--host 127.0.0.1] [--port 8000] [--author slinky] [--color "#3b82f6"]
    [--head default] [--tail default] [--move-log FILE] [--threads N]
    [--answer-by-ms 400] [--no-benchmark]
```

* `--snake NAME=RUN_DIR[:OPTS]` serves a snake at `/NAME`. Repeat the flag to serve several
  snakes. `RUN_DIR` holds `config.json` and `params.npz`, and the weights are read once at
  startup. The Python server's form `--agent NAME=az_modes:RUN_DIR[:OPTS]` is accepted too and
  means the same. The options are:
  * `simulations=N`: simulations per move. The default is 256, the faithful ladder search
    (bit-exact with Python's search).
  * `budget_ms=T`: stop the search T ms after the request reached the server. Without
    `simulations`, the cap is 16384 simulations, and the deadline is reached first. With both,
    the search stops at whichever comes first. A budgeted search is not the faithful mode: its
    moves depend on the machine's speed.

  The agent's name in logs and the move log is the agent text, for example
  `az_modes:/home/nathan/slinky-serve/az-modes-M-s0:simulations=256`, exactly as Python names
  it. The info's `version` is `slinky-native <agent text>` (Python: `slinky <agent text>`).
* `--move-log FILE` appends one JSON line per answer, with the same fields as Python's
  `MoveLedger`.
* `--threads N` sets the number of search threads. The default is the number of cores. Each
  `/move` runs its search on one thread, so N games are searched in parallel.
* `--answer-by-ms MS` answers with the safe move when the search is still running MS ms
  after the request arrived. The default is 400; 0 always waits. This is new: the Python
  server has no such deadline. It only fires under overload (all threads busy). The faithful
  256-simulation search takes 10 to 20 ms.
* `--no-benchmark` skips the startup self-benchmark (see "Seeds").

### Replacing the ladder bot

The current unit (`~/.config/systemd/user/slinky-snake.service`; this crate does not change it)
runs:

```text
ExecStart=/opt/slinky-venv/bin/python -m slinky.server --agent az-256=az_modes:/home/nathan/slinky-serve/az-modes-M-s0:simulations=256 --port 8000 --author nathanWolo --color "#ef4444" --head cosmic-horror --tail cosmic-horror --move-log /home/nathan/slinky-serve/logs/moves.jsonl
```

To switch the bot, copy the release binary to `/home/nathan/slinky-serve/bin/` and replace
that `ExecStart` with:

```text
ExecStart=/home/nathan/slinky-serve/bin/slinky-native-server --snake az-256=/home/nathan/slinky-serve/az-modes-M-s0:simulations=256 --port 8000 --author nathanWolo --color "#ef4444" --head cosmic-horror --tail cosmic-horror --move-log /home/nathan/slinky-serve/logs/moves.jsonl
```

The `Environment=JAX_PLATFORMS=cpu` and `PYTHONPATH` lines and the `WorkingDirectory` are no
longer needed, but they do no harm. The URL path (`/az-256`), the info fields, the
`127.0.0.1:8000` address behind the Cloudflare tunnel and the move log's format all stay the
same, so the tunnel and the Battlesnake registration need no change. Startup takes about half
a second: 1.6 ms to load the weights, then the 35-move benchmark. To spend the speed on a
stronger search, use `:simulations=1024`, or `:budget_ms=250` together with a simulation
cap.

## What it keeps from the Python server

| | Python (`server.py`) | native |
|---|---|---|
| routes | `GET /` lists the snakes, `GET /<name>` and `/<name>/` give the info, `POST /<name>/start\|move\|end`; anything else is 404 `{"error": "not found"}` | same, with the same path splitting (`//s`, `?query`) |
| bodies | `json.dumps` output: `{"move": "up"}`, `{}`, `{"snakes": ["/az-256"]}`, info `{"apiversion": "1", "author": ..., "color": ..., "head": ..., "tail": ..., "version": ...}` | byte for byte the same |
| body limits | `Content-Length` read with `int()`, 1 to 65536 bytes, else 400; not JSON, not an object, or a non-object `game` is 400 `{"error": ...}`; 5 s to send the request | same (the 400 message texts differ); a chunked body without a length is 400, as in Python |
| `/move` | never fails: a game it cannot play, or any exception, gets `safe_move` ("up" if even that raises) | same |
| `/start` | remembers `len(board.snakes)` per game id, at most 1,000 games (the oldest is evicted first); `/end` forgets the game | same, with Python's dict order and hashing (`1 == 1.0`, unhashable ids raise) |
| move log | `MoveLedger`: `t, snake, game, turn, ruleset, map, snakes, source, timeout, agent, move, ms, prev` (+ `wall`, `colo` from `Cf-Ray`, `end`), applied/missed/gap, a warning per miss, at most 1,000 games remembered | a port with the same fields, key order and number formats (`json.dumps`, `round(t, 3)`, `round(ms, 1)`), the same verdicts, and the same exceptions on junk; lines are written in order by a background thread |
| log lines | `%(asctime)s %(message)s`: `<snake> game <id> turn <t>: <move> (<agent>, <ms> ms)`, start, end won / lost or drew, the fallback, misses | the same lines (an exception is one line with its message instead of a traceback) |
| startup | precompiles four configs (minutes), benchmarks 20 duel moves and 5 per other mode | loads in 1.6 ms, benchmarks the same 35 positions (`data/examples.json`) |
| concurrency | a thread per connection, all moves sharing the GIL and JAX's CPU pool | a 2-thread tokio runtime for HTTP plus N search threads |

`ms` is the time to choose the move: the request conversion and the search, without
queueing. `wall` runs from the moment the request reached the handler to the moment the
answer was ready. Python measures `wall` after sending the response, which adds a fraction of
a millisecond.

### Seeds

The Python server searches move `i` of a snake with `fold_in(key(0), i)`. Here `i` counts from
1 over the life of the process, and it includes the 35 startup benchmark moves. The native
server keeps the same counter with the same rules. It advances on every `/move` that gets past
`check_sizes`, the game-id hash and `game_config` without an exception, including moves
answered with the safe move. It uses the same key. Started with the same snakes (and the
benchmark on), the two servers search the same requests with the same tie noise. With
`--no-benchmark` the counter starts at 0 instead of 35.

### Differences (all deliberate)

* **No compile step.** Python answers `safe_move` until it has compiled a game config it did
  not precompile, such as a royale with a non-default hazard damage or shrink period, or a
  3-snake game without `/start`. The native engine searches every game it can play from the
  first move.
* **No heuristic fallback.** Python plays its `heuristic` agent on boards other than 11x11 and
  in games with a config but no mode. Native answers with the safe move there. Ladder games
  never reach that case (see the workspace README, known gap 5).
* Where Python's own `safe_move` raises, the move is "up" in both, but Python names the agent
  `none` and native names it `safe_move`.
* A search still running after `--answer-by-ms` gets the safe move (Python has no deadline).
* Connections close after each response, as with Python's HTTP/1.0 handler.

## HTTP stack: hyper 1 on tokio

The stack is hyper 1 (`http1` server only) with `hyper-util` and `http-body-util`, on a
two-worker tokio runtime. These crates were already in the workspace lock file (the Lambda
crate pulls them in).

* **Slow clients cannot block a search thread.** Headers must arrive within 5 s
  (`header_read_timeout`), and the body is read asynchronously with a 5 s timeout and a 64 KiB
  limit, before any search thread is involved. With a blocking server such as `tiny_http`, the
  worker reads the body, and it has no read timeout. A handful of stalled uploads would then
  hold every worker and time out every game.
* **The search runs on its own threads.** A fixed pool (one thread per core) with a reusable
  search tree each, fed from a queue. The async side waits with a deadline (`--answer-by-ms`),
  so `/move` always answers in time even when the pool is saturated.
* hyper is the most widely deployed Rust HTTP implementation. It handles HTTP/1.1 framing
  (chunked bodies, pipelining, malformed requests) correctly, which a hand-written server
  would have to get right on its own.

`axum` would add routing that five routes do not need.

## Tests

`cargo test -p slinky-server-native` runs 11 unit tests and 6 HTTP tests (about 1 s):

* `ledger`: ports of `test_move_ledger_judges_each_answer` (verdicts, every field of a line,
  the exact `json.dumps` text) and `test_move_ledger_wraps_and_survives_junk` (wrapped edges,
  junk requests), plus Python's `int()` reads, floor arithmetic, and the exceptions on
  unhashable ids and a non-list `snakes`;
* `pyfmt`: `repr(float)`, `round`, `json.dumps` (escapes, NaN, big ints, nesting) and
  `str`/`repr`; `ordered`: dict order and eviction; `engine`: `--snake` parsing,
  `np.median`; `main`: command-line parsing; `examples`: the benchmark requests;
* `tests/http.rs` uses real HTTP on a free port. It needs the run dir `$NAZ_RUN` (default: the
  deployed M) and prints SKIP without it. It ports `test_http_endpoints`,
  `test_routes_are_per_snake`, `test_concurrent_moves` (32 at once) and
  `test_http_moves_are_logged`, and adds checks of the body limits (chunked, empty, over
  64 KiB, not UTF-8) and 501. It checks that the moves equal `Agent::search_game` with
  `fold_in(key(0), i)` for the server's `i`, including the counting rules for unplayable and
  oversized requests, and the 4 seats remembered at `/start` after a snake dies and forgotten
  at `/end`. It checks that a 60 ms budget answers quickly and that a search past
  `--answer-by-ms` is answered with the logged `safe_move`.

### Parity with the Python server over HTTP

`py/parity_http.py` starts the Python server (`slinky.server.main` with the deployed agent
`az-256=az_modes:<M>:simulations=256`, in-process) and the native binary with the same
arguments. It sends both the same requests one at a time and compares every response (status
and body bytes) and then the two move logs, line by line and field by field (all but `t`,
`ms` and `wall`). The requests are:

* real-shaped games in all seven ladder spellings of the four modes: standard 4p, duel, royale
  on the royale map, royale ruleset on the standard map, standard ruleset on the royale map,
  constrictor on the standard map, and constrictor on the empty map. They carry the ladder's
  settings and extra fields. The heuristic plays the other snakes and the servers' answers
  play ours. In about 5% of turns no request is sent (a gap), and in another 5% the answer is
  overridden (a miss);
* junk after each game: 400 bodies, wrapped, a 99x99 board, 122 food, unhashable, missing
  and integer game ids, a float board width, no turn, `{"board": 3}`, `{}`, a bad `/start` and
  `/end`, info and 404 routes.

```bash
bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \
    native/crates/server/py/parity_http.py \
    --native-bin /dev/shm/naz-target-server/release/slinky-native-server --games 3
```

Result (2026-10-15, seed 1, 4 rounds, games of up to 120 turns, M run dir, both servers
pinned to the same 4 cores, the native server with 2 search threads):

| check | count | result |
|---|---|---|
| requests compared | 1,596 | **0 mismatches** in status and body |
| game `/move`s (28 games: 4 of each spelling) | 868 | same move in all 868 (the same seeds: identical tie noise) |
| junk `/move`s (counter rules, fallbacks) | 280 | same answer, and the move counters stayed in step through them |
| `/start`, `/end`, routes, 400s | 448 | same status and body, except the 400 message text (140) and the info `version` (56), both expected |
| move-log lines | 1,176 / 1,176 | every field equal (`t`, `ms`, `wall` aside), and the text is identical with the numbers masked; verdicts applied 784, missed 41, gap 126, null 225, the same in both. The one expected difference is `agent` `none` (Python) vs `safe_move` (56 junk lines where Python's own `safe_move` raised) |
| median `ms` per move | | Python 34.2 ms (JAX, 4 cores), native 16.7 ms (one thread) |

An earlier run (seed 0, 7 games of up to 30 turns, 330 requests, 148 game moves) also gave
0 mismatches.

### Speed and load

This was measured with `py/load_test.py` against the release binary with 4 search threads
on 4 cores (`taskset -c 8-11`), on the PC while other jobs were running. The positions are
the benchmark positions of the four modes, and latency is wall time at the client:

| snake | concurrent games | median | p90 | max |
|---|---|---|---|---|
| `simulations=256` | 1 | 24.3 ms | 30.4 ms | 36.6 ms |
| `simulations=256` | 4 | 25.8 ms | 35.0 ms | 56.9 ms |
| `simulations=256` | 8 | 44.8 ms | 55.7 ms | 80.1 ms |
| `simulations=256` | 16 | 90.4 ms | 114.0 ms | 140.4 ms |
| `simulations=1024` | 1 / 4 | 77.5 / 82.6 ms | 82.5 / 101.6 ms | 95.8 / 159.0 ms |
| `budget_ms=250` | 1 / 4 | 250.6 / 250.9 ms | 250.7 / 253.7 ms | 264.6 / 254.0 ms |

Within its 250 ms, the budgeted search ran 2,081 to 6,563 simulations (median 3,244; one
thread per move). For a timed search, the log line also gives the simulation count, for
example `(az_modes:...:budget_ms=250, 250.0 ms, 3290 simulations)`.

`py/make_examples.py` regenerates `data/examples.json`, the benchmark positions, from
Python's `server.example_request`.
