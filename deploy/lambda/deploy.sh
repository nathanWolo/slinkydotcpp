#!/usr/bin/env bash
# Create or update the snake Lambda and its public Function URL (idempotent).
#
#   deploy/lambda/build.sh                      # package first
#   AWS_PROFILE=slinky deploy/lambda/deploy.sh  # region from the profile
#
# Creates, once: an execution role (logs only), a log group with 14-day
# retention, a private S3 bucket for the package (a direct upload of ~20 MB
# tends to be cut off; `aws s3 cp` uploads in parts with retries), the
# function (arm64, Python 3.13, MEMORY_MB = 1024 MB, 3 s timeout) and a Function
# URL without auth (Battlesnake's servers must be able to call it). Later runs
# only update the code and settings. Prints the URL and smoke-tests the native
# snake (GET /az-256, a few /move).
#
# The function's environment variables: the deployed ones are kept; the
# defaults below fill in what is missing; any of these set in this shell
# overrides the deployed value: SNAKE_AUTHOR, SNAKE_HEAD, SNAKE_TAIL (defaults
# slinky, default, default), SNAKE_SIMULATIONS_AZ_256 (256), NATIVE_MARGIN_MS
# (150), NATIVE_THREADS, and every SNAKE_COLOR_*, SNAKE_SIMULATIONS_* and
# SNAKE_BUDGET_MS_*. The AWS CLI is $AWS, else `aws` on the PATH, else
# ~/.local/bin/aws.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
AWS="${AWS:-$(command -v aws || echo "$HOME/.local/bin/aws")}"
profile="${AWS_PROFILE:-slinky}"
region="$("$AWS" configure get region --profile "$profile")"
fn="${FUNCTION_NAME:-slinky-snake}"
role="$fn-lambda"
zip="$here/dist/snake.zip"
memory="${MEMORY_MB:-1024}" # 1769 MB is one full vCPU (1024 MB is 0.58 of one)

aws() { "$AWS" --profile "$profile" --region "$region" "$@"; }
[ -f "$zip" ] || { echo "no $zip: run build.sh first" >&2; exit 1; }
python3 -c "import sys, zipfile; n = set(zipfile.ZipFile(sys.argv[1]).namelist()); \
sys.exit(not {'bin/slinky-engine', 'weights/az-modes/params.npz'} <= n)" "$zip" \
    || { echo "$zip has no native engine: rebuild it with build.sh" >&2; exit 1; }

account="$(aws sts get-caller-identity --query Account --output text)"
echo "account $account, region $region, function $fn"

if ! aws iam get-role --role-name "$role" >/dev/null 2>&1; then
    echo "creating role $role"
    aws iam create-role --role-name "$role" --output text --query Role.Arn \
        --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
    aws iam attach-role-policy --role-name "$role" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi
role_arn="arn:aws:iam::$account:role/$role"

bucket="$fn-deploy-$account-$region"
if ! aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
    echo "creating bucket $bucket"
    aws s3api create-bucket --bucket "$bucket" \
        --create-bucket-configuration LocationConstraint="$region" >/dev/null
    aws s3api put-public-access-block --bucket "$bucket" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
aws s3 cp --only-show-errors "$zip" "s3://$bucket/snake.zip"
code=(--s3-bucket "$bucket" --s3-key snake.zip)

log_group="/aws/lambda/$fn"
if [ -z "$(aws logs describe-log-groups --log-group-name-prefix "$log_group" \
        --query "logGroups[?logGroupName=='$log_group'].logGroupName" --output text)" ]; then
    aws logs create-log-group --log-group-name "$log_group"
fi
aws logs put-retention-policy --log-group-name "$log_group" --retention-in-days 14

# The environment: deployed values, defaults for what is missing, this shell's on top.
exists=false
deployed="{}"
if aws lambda get-function --function-name "$fn" >/dev/null 2>&1; then
    exists=true
    deployed="$(aws lambda get-function-configuration --function-name "$fn" \
        --query 'Environment.Variables' --output json)"
fi
env_file="$(mktemp)"
trap 'rm -f "$env_file"' EXIT
DEPLOYED="$deployed" python3 - "$env_file" <<'PY'
import json
import os
import sys

deployed = json.loads(os.environ["DEPLOYED"] or "null") or {}
merged = {
    "SNAKE_AUTHOR": "slinky",
    "SNAKE_HEAD": "default",
    "SNAKE_TAIL": "default",
    "SNAKE_SIMULATIONS_AZ_256": "256",
    "NATIVE_MARGIN_MS": "150",
}
merged.update(deployed)
for k, v in os.environ.items():
    if k in ("SNAKE_AUTHOR", "SNAKE_HEAD", "SNAKE_TAIL", "NATIVE_MARGIN_MS", "NATIVE_THREADS") or (
        k.startswith(("SNAKE_COLOR_", "SNAKE_SIMULATIONS_", "SNAKE_BUDGET_MS_"))
    ):
        merged[k] = v
with open(sys.argv[1], "w") as f:
    json.dump({"Variables": merged}, f)
print("environment:", json.dumps(merged, sort_keys=True))
PY

settings=(--runtime python3.13 --handler lambda_function.lambda_handler
          --memory-size "$memory" --timeout 3 --environment "file://$env_file")
if $exists; then
    echo "updating $fn"
    aws lambda update-function-code --function-name "$fn" "${code[@]}" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
    aws lambda update-function-configuration --function-name "$fn" "${settings[@]}" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
else
    echo "creating $fn"
    # A new role takes a few seconds before Lambda can assume it.
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if aws lambda create-function --function-name "$fn" --architectures arm64 \
                --role "$role_arn" --code "S3Bucket=$bucket,S3Key=snake.zip" "${settings[@]}" \
                >/dev/null 2>"$here/dist/create.err"; then
            break
        fi
        grep -q "cannot be assumed" "$here/dist/create.err" || { cat "$here/dist/create.err" >&2; exit 1; }
        [ "$attempt" = 10 ] && { cat "$here/dist/create.err" >&2; exit 1; }
        echo "waiting for the role to propagate ($attempt)"
        sleep 5
    done
    aws lambda wait function-active-v2 --function-name "$fn"
fi

# Cap concurrency so a flood of requests can't run up a bill. New accounts
# whose total concurrency quota is only 10 can't reserve any (and are capped
# at 10 anyway).
aws lambda put-function-concurrency --function-name "$fn" --reserved-concurrent-executions 10 \
    >/dev/null 2>&1 || echo "note: could not reserve concurrency (account quota too low); not capped per function"

if ! aws lambda get-function-url-config --function-name "$fn" >/dev/null 2>&1; then
    echo "creating the Function URL"
    aws lambda create-function-url-config --function-name "$fn" --auth-type NONE >/dev/null
    aws lambda add-permission --function-name "$fn" --statement-id public-url \
        --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
    aws lambda add-permission --function-name "$fn" --statement-id public-url-invoke \
        --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null
fi
url="$(aws lambda get-function-url-config --function-name "$fn" --query FunctionUrl --output text)"
echo "snake URLs: ${url%/}/<name>, one per name in lambda_function.SNAKES and NATIVE"

# Smoke test: the native snake's info, then /move of a 4-snake standard game (the first
# may be a cold start, which also starts the engine process).
base="${url%/}/az-256"
echo "GET $base: $(curl -sS --max-time 10 "$base")"
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
out=""
for _ in 1 2 3; do
    out="$(curl -sS --max-time 10 -H 'Content-Type: application/json' -d "$move" \
        -w ' (%{time_total} s)' "$base/move")"
    echo "POST $base/move: $out"
done
echo "$out" | grep -qE '"move": ?"(up|down|left|right)"' \
    || { echo "smoke test failed: no move from $base/move" >&2; exit 1; }
