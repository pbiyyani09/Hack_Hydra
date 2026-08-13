#!/usr/bin/env bash
# Start local HydraDB. Creates the data dirs and auth token on first run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p data/store data/cache

if [[ ! -f data/auth-token ]]; then
  printf '%s\n' 'local-development-token-32-bytes' > data/auth-token
  chmod 600 data/auth-token
  echo "created data/auth-token"
fi

# Compose interpolates ${UID}/${GID} from this file. They cannot simply be
# exported, because bash makes UID readonly.
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" > .env

docker compose up -d

printf 'waiting for hydradb'
for _ in $(seq 1 60); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9090/readyz || true)" == "200" ]]; then
    echo " ready"
    echo "  bolt   neo4j://127.0.0.1:7687"
    echo "  http   http://127.0.0.1:8443/v1/graphs/default/query"
    echo "  admin  http://127.0.0.1:9090/{readyz,livez,metrics}"
    exit 0
  fi
  printf '.'
  sleep 1
done

echo " TIMED OUT"
docker compose logs --tail 40
exit 1
