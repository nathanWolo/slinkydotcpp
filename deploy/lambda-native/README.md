# The native snakes as their own Lambda function

`slinky-native` (arm64, `provided.al2023`, 1769 MB = one vCPU) runs the Rust handler of
`native/crates/lambda` (binary `slinky-lambda` as `bootstrap`, settings in that crate's README)
with the az_modes weights. One Function URL, one path per snake:

| path | search |
|---|---|
| `/az-256` | 256 simulations (the faithful port of the ladder agent); arena name AlphaSlinky-256 |
| `/az-t200` | as many simulations as fit in 200 ms from the Function URL's receipt of the request (cap 16,384); arena name AlphaSlinky |

Why a function of its own: the Python function `slinky-snake` (`deploy/lambda`) takes ~410 ms
to start a container (Python and numpy), and every container it starts in the middle of a game
(concurrent games scale it out) answered past the arena's 500 ms: 3 timeouts in its first
1,346 ladder moves. A Rust runtime starts in tens of ms. Lambda bills per request and duration,
so a second function costs nothing more.

    deploy/lambda-native/build.sh                         # dist/native.zip (deploy.sh runs it)
    AWS_PROFILE=slinky deploy/lambda-native/deploy.sh     # create or update, URL, smoke test

Change the line-up or the think time with `SNAKES` (e.g.
`SNAKES=az-256=az-modes:simulations=256,az-t250=az-modes:budget_ms=250`, plus that snake's
`SNAKE_COLOR_`/`SNAKE_HEAD_`/`SNAKE_TAIL_` variables). Each move logs its `ms`, `sims`,
`deadline_ms` and `cut` to CloudWatch (`/aws/lambda/slinky-native`).
