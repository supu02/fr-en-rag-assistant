#!/usr/bin/env bash
# Build the Docker image and deploy it to AWS Lambda with AWS SAM.
#
# Needs: Docker Desktop running, the AWS CLI signed in (`aws login`), the SAM CLI.
# Deploys to the Region of your AWS CLI profile (`aws configure get region`).
# No API key: the function answers with a Mistral model on Amazon Bedrock, using its own
# IAM role (see template.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
export SAM_CLI_TELEMETRY=0  # don't send usage statistics to AWS

# Check the three things that usually go wrong before spending 10 minutes on a build.
REGION=${AWS_REGION:-$(aws configure get region || true)}
if [ -z "$REGION" ]; then
  echo "No AWS Region set. Use your project's Region, for example: aws configure set region eu-north-1"
  exit 1
fi
if ! ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null); then
  echo "The AWS CLI is not signed in. Run: aws login"
  exit 1
fi
# Docker Desktop installed for one user keeps its command in ~/.docker/bin and its socket
# in ~/.docker/run. Point DOCKER_HOST at that socket so the SAM CLI finds it too.
if ! command -v docker >/dev/null 2>&1 && [ -x "$HOME/.docker/bin/docker" ]; then
  PATH="$HOME/.docker/bin:$PATH"
fi
if [ -z "${DOCKER_HOST:-}" ] && [ ! -S /var/run/docker.sock ] && [ -S "$HOME/.docker/run/docker.sock" ]; then
  export DOCKER_HOST="unix://$HOME/.docker/run/docker.sock"
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then run this script again."
  exit 1
fi
echo "Deploying to AWS account ${ACCOUNT}, Region ${REGION}."

sam build
sam deploy --region "$REGION"

# Warm-up: the first start after a deployment is slow (about 2 minutes on 19 Nov 2026, against
# about 10 s later), because Lambda reads the new image's files for the first time. Make that
# wait here rather than for the first visitor.
STACK=$(sed -n 's/^stack_name *= *"\(.*\)"/\1/p' samconfig.toml)
URL=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
echo "Waking up the new version at ${URL} (can take about 2 minutes)..."
for attempt in 1 2 3; do
  # If the first start runs past the function's timeout, Lambda starts again; the second
  # attempt is faster because part of the image is already cached.
  if curl -sf --max-time 150 -o /dev/null -w "Ready: /health answered in %{time_total} s.\n" "${URL%/}/health"; then
    exit 0
  fi
  echo "Not ready yet (attempt ${attempt} of 3)."
done
echo "The site did not answer. Check the logs: sam logs --stack-name ${STACK} --tail"
exit 1
