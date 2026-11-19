#!/usr/bin/env bash
# Build the Docker image and deploy it to AWS Lambda with AWS SAM.
#
# Needs: Docker Desktop running, the AWS CLI signed in (`aws sso login`), the SAM CLI.
# The Mistral key is read from .env: never typed on the command line, never committed.
# Without a key, the deployed API still serves /search; /ask returns 503.
set -euo pipefail
cd "$(dirname "$0")/.."

KEY=""
if [ -f .env ]; then
  KEY=$(grep -E '^MISTRAL_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d "\"'" || true)
fi
if [ -z "$KEY" ]; then
  echo "No MISTRAL_API_KEY in .env: deploying search only (/ask will return 503)."
fi

sam build
sam deploy --parameter-overrides "MistralApiKey=${KEY}"
