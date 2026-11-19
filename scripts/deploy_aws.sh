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
