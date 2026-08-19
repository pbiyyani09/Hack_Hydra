#!/usr/bin/env bash
# Boot HydraDB OSS for MedMemGraph and block until /readyz is 200.
#
# Referenced as step 1 of the README Quickstart. Two pins here are deliberate
# and are not stylistic choices:
#
#   * Image `0.1.1`, never `latest` and never `0.1.0`. `main` has no CI gate,
#     so `latest` can move under you mid-sprint; `0.1.0` is amd64-only.
#   * `CLOUD_PROVIDER=memory`, never `local`. `local` is reported unsafe under
#     sustained write load upstream, and corpus ingest IS sustained write load.
#     The cost of `memory` is that the graph does not survive a container
#     restart — acceptable here because llm.py caches every completion to disk
#     and every node id is a deterministic SHA-256 fold, so a re-ingest is
#     idempotent and costs ~$0 in API spend.
#
set -euo pipefail

IMAGE="ghcr.io/hydra-db/hydradb:0.1.1"
NAME="${HYDRADB_CONTAINER_NAME:-hydradb}"
TOKEN_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/auth-token"

# The token is a shared secret with the client, not a public value. It must be
# >=32 chars and must not be "change-me", or the container refuses to boot.
# Gitignored at the repo root (/auth-token).
if [ ! -f "$TOKEN_FILE" ]; then
  printf '%s\n' 'local-development-token-32-bytes' > "$TOKEN_FILE"
  chmod 644 "$TOKEN_FILE"
  echo "created $TOKEN_FILE"
fi

if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
  echo "removing existing container '${NAME}'"
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" \
  -p 7687:7687 -p 8443:8443 -p 9090:9090 \
  -v "$TOKEN_FILE:/auth-token:ro" \
  -e CLOUD_PROVIDER=memory \
  -e GRAPH_AUTH_TOKEN_FILE=/auth-token \
  -e GRAPH_ALLOW_PLAINTEXT=true \
  -e GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
  -e RUST_MIN_STACK=33554432 \
  "$IMAGE" >/dev/null

printf 'waiting for hydradb'
for _ in $(seq 1 60); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9090/readyz || true)" = "200" ]; then
    echo " ready"
    echo "  bolt   bolt://127.0.0.1:7687"
    echo "  http   http://127.0.0.1:8443/v1/graphs/default/query"
    echo "  admin  http://127.0.0.1:9090/{readyz,livez,metrics}"
    echo
    echo "Export these (or put them in a gitignored .env):"
    echo "  export HYDRA_AUTH_TOKEN=$(cat "$TOKEN_FILE")"
    echo "  export HYDRA_BOLT_URI=bolt://127.0.0.1:7687"
    exit 0
  fi
  printf '.'
  sleep 1
done

echo " TIMED OUT"
docker logs --tail 40 "$NAME"
exit 1
