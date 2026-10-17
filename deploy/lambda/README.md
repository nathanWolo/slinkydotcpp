# AWS Lambda snake

One Lambda function (`slinky-snake`, arm64, Python 3.13) behind one Function URL, one path per
snake:

- the search-free trained networks (`ppo`, `reinforce`, `dqn`, `rainbow`, `az-0`) ported to
  plain NumPy (`snake_numpy.py`). Each plays the 11×11 duel. `tests/test_snake_numpy.py`
  checks the port against the JAX agents.
- the native AlphaZero snake `az-256`: the multi-mode az_modes M network with its 256-simulation
  search (the ladder agent, all four ladder modes: standard, duel, royale, constrictor), played
  by the Rust engine `bin/slinky-engine` (`native/crates/lambda`, `src/bin/engine.rs`). The
  handler starts the engine once per container, on first use (usually the game's `/start`), and
  talks to it in JSON lines. `tests/test_lambda_native.py` checks the handler with a local
  build of the engine.

Files:

- `lambda_function.py`: the handler (`GET /<snake>`, `POST /<snake>/start|move|end`). A move
  request always gets a move; on any error it falls back to a safe move. For a native snake,
  the engine's search stops at the request's deadline, `game.timeout` (500 ms when missing) less
  `NATIVE_MARGIN_MS`, counted from the handler's start; if the engine has not answered 30 ms
  after that, has died or answers nonsense, the move is the safe move and the engine is
  restarted on the next request.
- `package_checkpoints.py`: copies the checkpoints the handler names from
  `baselines/checkpoints/` (without the noisy layers' scales).
- `build.sh [python]`: builds `dist/snake.zip` (arm64, Python 3.13, numpy 2.2.6 wheel, the
  engine cross-built for aarch64 with cargo-lambda and `panic = unwind`, the az-256 weights as
  `weights/az-modes/`); the python needs pip and numpy. `ENGINE=<prebuilt aarch64
  slinky-engine>` skips the Rust build; `NATIVE_RUN` is the az_modes run dir (default
  `/home/nathan/slinky-serve/az-modes-M-s0`).
- `deploy.sh`: creates or updates the function, its role, log group, package bucket and a
  public Function URL, using the AWS CLI profile `AWS_PROFILE` (default `slinky`) and the CLI
  `$AWS` (else `aws` on the PATH, else `~/.local/bin/aws`). It keeps the function's deployed
  environment variables, adds the missing defaults and takes overrides from the shell (see its
  header); `MEMORY_MB` sets the memory (default 1024). It ends with a smoke test of `/az-256`. It
  creates AWS resources in your account and the URL is public (Battlesnake's servers must be
  able to call it); read it before running it.
- `check_live.py <url>/<snake>`: plays duels against a deployed snake through the real API and
  compares each remote move with the local NumPy network, or for `az-256` with a local
  `slinky-engine` (host build); prints the round-trip latency (median, p95, p99, max).

Settings (environment variables of the function): `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL`
and `SNAKE_COLOR_<NAME>` (e.g. `SNAKE_COLOR_PPO`, `SNAKE_COLOR_AZ_256`) for the info
responses; `CHECKPOINTS`. Native snakes: `SNAKE_SIMULATIONS_<NAME>` (simulations per move,
default 256 for `az-256`), `SNAKE_BUDGET_MS_<NAME>` (optional: also stop that many ms after
the arrival), `NATIVE_MARGIN_MS` (default 150), `NATIVE_THREADS` (root-parallel trees, default
1), `WEIGHTS_DIR`, `ENGINE_BIN`.

```bash
deploy/lambda/build.sh                          # or build.sh <python with pip and numpy>
AWS_PROFILE=slinky deploy/lambda/deploy.sh      # prints the URL, smoke-tests /az-256
python deploy/lambda/check_live.py https://<id>.lambda-url.<region>.on.aws/ppo --games 8
python deploy/lambda/check_live.py https://<id>.lambda-url.<region>.on.aws/az-256 --games 4 \
    --engine <x86 slinky-engine> --native-run /home/nathan/slinky-serve/az-modes-M-s0
```

Register `https://<id>.lambda-url.<region>.on.aws/<snake>` as the Battlesnake's URL.

Tests (no AWS): `tests/test_lambda_native.py` runs `lambda_function.lambda_handler` with an x86
build of the engine (`CARGO_PROFILE_RELEASE_PANIC=unwind cargo build --release -p
slinky-lambda --bin slinky-engine`, path in `$SLINKY_ENGINE_BIN`): `/az-256` info, start, move
and end in the four modes (moves equal the engine's own), malformed requests, 200 ms and
500 ms `game.timeout` with a search far too long for them, the engine killed and frozen
mid-game, a missing or broken engine, and the numpy snakes' answers unchanged.
