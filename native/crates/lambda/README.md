# slinky-lambda

AWS Lambda handler (custom runtime `provided.al2023`, arm64) for the native
AlphaZero snakes. It sits next to the Python deployment in `deploy/lambda/` and
routes the same way behind one Function URL:

    GET  /<snake>          info (apiversion, author, color, head, tail, version)
    POST /<snake>/start    remembers the game's snake count (bounded per-process cache)
    POST /<snake>/move     {"move": "up|down|left|right"}; any error answers "up"
    POST /<snake>/end      logged

Environment: `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL`, `SNAKE_COLOR_<NAME>`
(info response), `WEIGHTS_DIR` (default `/var/task/weights/az-modes`).

The snake names are the `SNAKES` table in `src/main.rs` (placeholder: `slinky`).
The engine plugs in at one place, `engine_choose_move` in `src/main.rs`
(`choose_move(request_json, start_snakes) -> Result<String, Error>` from the
search crate; the `slinky-search` dependency and `engine` feature are stubbed
in `Cargo.toml`). Until then it answers "up".

## Toolchain (no sudo)

Rust via rustup (`source native/.cargo-env.sh`), plus `cargo-lambda`, which
cross-links with zig. Both come from a throwaway venv:

    uv venv ~/.local/lambda-tools && uv pip install --python ~/.local/lambda-tools/bin/python cargo-lambda
    export PATH=$HOME/.local/lambda-tools/bin:$PATH
    rustup target add aarch64-unknown-linux-gnu

(the `cargo-lambda` wheel bundles the binary and depends on `ziglang`).

## Build and package

    source native/.cargo-env.sh
    export CARGO_TARGET_DIR=/dev/shm/naz-target-lambda PATH=$HOME/.local/lambda-tools/bin:$PATH
    # stage the weights as <stage>/weights/az-modes/{config.json,params.npz}; --include is
    # relative to the current directory and keeps that relative path inside the zip
    cd <stage> && cargo lambda build --release --arm64 \
        --manifest-path <repo>/native/crates/lambda/Cargo.toml --output-format zip --include weights

Output: `$CARGO_TARGET_DIR/lambda/slinky-lambda/bootstrap.zip` (`bootstrap`
at the root, `weights/az-modes/...` which is `/var/task/weights/az-modes`).
Placeholder build: binary 1.6 MB, zip 2.0 MB with the 1.3 MB M model.

## Local test (x86)

    cargo lambda watch -p slinky-lambda --release -P 9001      # builds for the host
    curl localhost:9001/slinky
    curl -XPOST localhost:9001/slinky/move -d '{"game":{"id":"g"},"board":{"snakes":[]}}'

## Deploy outline (not run: needs AWS credentials)

Mirror `deploy/lambda/deploy.sh` (role, log group with 14-day retention, S3
bucket for the package, public Function URL, reserved concurrency 10) with
these differences: `--runtime provided.al2023 --architectures arm64`,
`--handler bootstrap`, the zip above, function name e.g. `slinky-snake-native`
(separate from `slinky-snake`), and memory/timeout sized for the search (start
at 1024 MB, 3 s; the Battlesnake limit is 500 ms per move). Set the
environment `Variables={SNAKE_AUTHOR=...,SNAKE_HEAD=...,SNAKE_TAIL=...}`.
Alternatively `cargo lambda deploy slinky-lambda --binary-name slinky-lambda
--enable-function-url` with the same profile.
