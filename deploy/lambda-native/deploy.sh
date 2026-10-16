#!/usr/bin/env bash
# Deploy the native snakes as their own Lambda function (default slinky-native, arm64,
# provided.al2023: the Rust binary is the runtime, so a new container starts in tens of ms
# instead of the Python function's ~400 ms) with a public Function URL. Builds the zip first
# (build.sh); SKIP_BUILD=1 deploys dist/native.zip as it is.
#
#   AWS_PROFILE=slinky deploy/lambda-native/deploy.sh
#
# Settings (this shell's value wins, then the deployed one, then the default): FUNCTION
# (slinky-native), REGION (us-east-2), ROLE (slinky-snake-lambda, shared with the Python
# function), MEMORY_MB (1769: one full vCPU), and the function's environment: SNAKES
# (az-256=az-modes:simulations=256,az-t200=az-modes:budget_ms=200), SNAKE_AUTHOR (slinky),
# SNAKE_HEAD / SNAKE_TAIL (default), DEADLINE_MARGIN_MS (150), THREADS, and every
# SNAKE_COLOR_*, SNAKE_HEAD_*, SNAKE_TAIL_* (az-t200: #ef4444, cosmic-horror; az-256:
# #166534, bendr, hook). A deployed value wins over these defaults: to change one, set it here.
# The AWS CLI is $AWS, else `aws` on the PATH, else ~/.local/bin/aws.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
AWS="${AWS:-$(command -v aws || echo "$HOME/.local/bin/aws")}"
fn="${FUNCTION:-slinky-native}"
region="${REGION:-us-east-2}"
role="${ROLE:-slinky-snake-lambda}"
memory="${MEMORY_MB:-1769}"
aws() { "$AWS" --region "$region" "$@"; }

[ "${SKIP_BUILD:-}" = 1 ] || "$here/build.sh"
zip="$here/dist/native.zip"
[ -f "$zip" ] || { echo "no $zip: run build.sh first" >&2; exit 1; }

account="$(aws sts get-caller-identity --query Account --output text)"
echo "account $account, region $region, function $fn"
role_arn="$(aws iam get-role --role-name "$role" --query Role.Arn --output text)" \
    || { echo "no role $role: deploy the Python function first (deploy/lambda/deploy.sh creates it)" >&2; exit 1; }

log_group="/aws/lambda/$fn"
if [ -z "$(aws logs describe-log-groups --log-group-name-prefix "$log_group" \
        --query "logGroups[?logGroupName=='$log_group'].logGroupName" --output text)" ]; then
    aws logs create-log-group --log-group-name "$log_group"
    aws logs put-retention-policy --log-group-name "$log_group" --retention-in-days 30
fi

exists=false
deployed=""
if aws lambda get-function --function-name "$fn" >/dev/null 2>&1; then
    exists=true
    deployed="$(aws lambda get-function-configuration --function-name "$fn" \
        --query 'Environment.Variables' --output json)"
fi
env_file="$(mktemp)"
trap 'rm -f "$env_file"' EXIT
DEPLOYED="$deployed" python3 - "$env_file" <<'PY'
import json, os, sys
merged = {
    "SNAKES": "az-256=az-modes:simulations=256,az-t200=az-modes:budget_ms=200",
    "SNAKE_AUTHOR": "slinky",
    "SNAKE_HEAD": "default",
    "SNAKE_TAIL": "default",
    "DEADLINE_MARGIN_MS": "150",
}
merged |= {"SNAKE_COLOR_AZ_T200": "#ef4444", "SNAKE_HEAD_AZ_T200": "cosmic-horror",
           "SNAKE_TAIL_AZ_T200": "cosmic-horror",
           "SNAKE_COLOR_AZ_256": "#166534", "SNAKE_HEAD_AZ_256": "bendr", "SNAKE_TAIL_AZ_256": "hook"}
merged.update(json.loads(os.environ["DEPLOYED"] or "null") or {})
for k, v in os.environ.items():
    if k in ("SNAKES", "SNAKE_AUTHOR", "SNAKE_HEAD", "SNAKE_TAIL", "DEADLINE_MARGIN_MS", "THREADS") or (
        k.startswith(("SNAKE_COLOR_", "SNAKE_HEAD_", "SNAKE_TAIL_"))
    ):
        merged[k] = v
with open(sys.argv[1], "w") as f:
    json.dump({"Variables": merged}, f)
print("environment:", json.dumps(merged, sort_keys=True))
PY

settings=(--runtime provided.al2023 --handler bootstrap --memory-size "$memory" --timeout 3
          --environment "file://$env_file")
if $exists; then
    echo "updating $fn"
    aws lambda update-function-code --function-name "$fn" --zip-file "fileb://$zip" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
    aws lambda update-function-configuration --function-name "$fn" "${settings[@]}" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
else
    echo "creating $fn"
    aws lambda create-function --function-name "$fn" --architectures arm64 --role "$role_arn" \
        --zip-file "fileb://$zip" "${settings[@]}" >/dev/null
    aws lambda wait function-active-v2 --function-name "$fn"
fi

# Cap concurrency so a flood of requests can't run up a bill.
aws lambda put-function-concurrency --function-name "$fn" --reserved-concurrent-executions 10 \
    >/dev/null 2>&1 || echo "note: could not reserve concurrency (account quota too low); not capped per function"

if ! aws lambda get-function-url-config --function-name "$fn" >/dev/null 2>&1; then
    echo "creating the Function URL"
    aws lambda create-function-url-config --function-name "$fn" --auth-type NONE >/dev/null
    aws lambda add-permission --function-name "$fn" --statement-id public-url \
        --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
    aws lambda add-permission --function-name "$fn" --statement-id public-url-invoke \
        --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null
    sleep 5  # a new URL's permissions take a moment
fi
url="$(aws lambda get-function-url-config --function-name "$fn" --query FunctionUrl --output text)"
echo "snake URLs: ${url%/}/<name>, one per name in SNAKES"

# Smoke test: each snake's info, then three /move of a 4-snake standard game (the first one
# may be a cold start).
move="$(python3 -c '
import json
cells = {"a": (1, 1), "b": (9, 9), "c": (1, 9), "d": (9, 1)}
snakes = [{"id": k, "name": k, "health": 100, "length": 3, "head": {"x": x, "y": y},
           "body": [{"x": x, "y": y}] * 3} for k, (x, y) in cells.items()]
print(json.dumps({"game": {"id": "deploy-smoke", "ruleset": {"name": "standard"},
                           "map": "standard", "timeout": 500, "source": "custom"},
                  "turn": 0, "you": snakes[0],
                  "board": {"width": 11, "height": 11, "food": [{"x": 5, "y": 5}],
                            "hazards": [], "snakes": snakes}}))')"
for name in az-256 az-t200; do
    base="${url%/}/$name"
    echo "GET $base: $(curl -sS --max-time 10 "$base")"
    out=""
    for _ in 1 2 3; do
        out="$(curl -sS --max-time 10 -H 'Content-Type: application/json' -d "$move" \
            -w ' (%{time_total} s)' "$base/move")"
        echo "POST $base/move: $out"
    done
    echo "$out" | grep -qE '"move": ?"(up|down|left|right)"' \
        || { echo "smoke test failed: no move from $base/move" >&2; exit 1; }
done
