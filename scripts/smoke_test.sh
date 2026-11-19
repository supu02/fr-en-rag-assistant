#!/usr/bin/env bash
# Call a running API (local container, CI or the deployed URL) and check the answers.
# Usage: ./scripts/smoke_test.sh http://localhost:8000
set -euo pipefail
URL="${1%/}"

for i in $(seq 1 60); do curl -sf "$URL/health" > /dev/null && break; sleep 2; done
curl -sf "$URL/health"; echo
curl -sf -X POST "$URL/search" -H 'Content-Type: application/json' \
  -d '{"question": "Quel préavis pour quitter un logement meublé ?"}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin)['results']; assert r; print('FR top result:', r[0]['page_title'])"
curl -sf -X POST "$URL/search" -H 'Content-Type: application/json' \
  -d '{"question": "How much notice must I give to leave a furnished flat?"}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin)['results']; assert r; print('EN top result:', r[0]['page_title'])"
