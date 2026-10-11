#!/usr/bin/env bash
# Create or update the snake Lambda and its public Function URL (idempotent).
#
#   deploy/lambda/build.sh                      # package first
#   AWS_PROFILE=slinky deploy/lambda/deploy.sh  # region from the profile
#
# Creates, once: an execution role (logs only), a log group with 14-day
# retention, the function (arm64, Python 3.13, 1024 MB, 3 s timeout) and a
# Function URL without auth (Battlesnake's servers must be able to call it).
# Later runs only update the code and settings. Prints the URL.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
profile="${AWS_PROFILE:-slinky}"
region="$(command aws configure get region --profile "$profile")"
fn="${FUNCTION_NAME:-slinky-snake}"
role="$fn-lambda"
zip="$here/dist/snake.zip"
env_vars="Variables={SNAKE_AUTHOR=${SNAKE_AUTHOR:-slinky},SNAKE_COLOR=${SNAKE_COLOR:-#3b82f6},SNAKE_HEAD=${SNAKE_HEAD:-default},SNAKE_TAIL=${SNAKE_TAIL:-default}}"

aws() { command aws --profile "$profile" --region "$region" "$@"; }
[ -f "$zip" ] || { echo "no $zip: run build.sh first" >&2; exit 1; }

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

log_group="/aws/lambda/$fn"
if [ -z "$(aws logs describe-log-groups --log-group-name-prefix "$log_group" \
        --query "logGroups[?logGroupName=='$log_group'].logGroupName" --output text)" ]; then
    aws logs create-log-group --log-group-name "$log_group"
fi
aws logs put-retention-policy --log-group-name "$log_group" --retention-in-days 14

settings=(--runtime python3.13 --handler lambda_function.lambda_handler
          --memory-size 1024 --timeout 3 --environment "$env_vars")
if aws lambda get-function --function-name "$fn" >/dev/null 2>&1; then
    echo "updating $fn"
    aws lambda update-function-code --function-name "$fn" --zip-file "fileb://$zip" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
    aws lambda update-function-configuration --function-name "$fn" "${settings[@]}" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$fn"
else
    echo "creating $fn"
    # A new role takes a few seconds before Lambda can assume it.
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if aws lambda create-function --function-name "$fn" --architectures arm64 \
                --role "$role_arn" --zip-file "fileb://$zip" "${settings[@]}" >/dev/null 2>"$here/dist/create.err"; then
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
echo "snake URL: ${url%/}/ppo"
