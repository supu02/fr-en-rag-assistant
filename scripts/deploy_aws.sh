#!/usr/bin/env bash
# Build the Docker image and deploy it to AWS Lambda with AWS SAM.
#
# Needs: Docker Desktop running, the AWS CLI signed in (`aws login`), the SAM CLI.
# Deploys to the Region of your AWS CLI profile (`aws configure get region`).
# The Mistral key is read from .env: never typed on the command line, never committed.
# Without a key, the deployed API still serves /search; /ask returns 503.
set -euo pipefail
cd "$(dirname "$0")/.."

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
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then run this script again."
  exit 1
fi
echo "Deploying to AWS account ${ACCOUNT}, Region ${REGION}."

KEY=""
if [ -f .env ]; then
  KEY=$(grep -E '^MISTRAL_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d "\"'" || true)
fi
if [ -z "$KEY" ]; then
  echo "No MISTRAL_API_KEY in .env: deploying search only (/ask will return 503)."
fi

sam build
sam deploy --region "$REGION" --parameter-overrides "MistralApiKey=${KEY}"
