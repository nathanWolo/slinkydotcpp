# AWS Lambda snake

The search-free trained networks (`ppo`, `reinforce`, `dqn`, `rainbow`, `az-0`)
ported to plain NumPy (`snake_numpy.py`) and served behind one Lambda Function
URL, one path per snake. Each plays the 11×11 duel. `tests/test_snake_numpy.py`
checks the port against the JAX agents.

- `lambda_function.py`: the handler (`GET /<snake>`, `POST /<snake>/start|move|end`).
  A move request always gets a move; on any error it falls back to a safe move.
  Settings by environment variable: `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL`,
  `SNAKE_COLOR_<NAME>`, `CHECKPOINTS`.
- `package_checkpoints.py`: copies the checkpoints the handler names from
  `baselines/checkpoints/` (without the noisy layers' scales).
- `build.sh [python]`: builds `dist/snake.zip` (arm64, Python 3.13, numpy
  2.2.6 wheel); the python needs pip and numpy.
- `deploy.sh`: creates or updates the function, its role, log group, package
  bucket and a public Function URL, using the AWS CLI profile `AWS_PROFILE`
  (default `slinky`). It creates AWS resources in your account and the URL is
  public (Battlesnake's servers must be able to call it); read it before
  running it.
- `check_live.py <url>/<snake>`: plays games against a deployed snake through
  the real API and compares each remote move with the local NumPy network.

```bash
deploy/lambda/build.sh
AWS_PROFILE=slinky deploy/lambda/deploy.sh      # prints the URL
python deploy/lambda/check_live.py https://<id>.lambda-url.<region>.on.aws/ppo --games 8
```

Register `https://<id>.lambda-url.<region>.on.aws/<snake>` as the Battlesnake's URL.
