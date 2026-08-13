#!/usr/bin/env bash
# Verify a local HydraDB is up and exercises the features we depend on.
# Safe to re-run: it writes into an isolated id range (9000+).
set -uo pipefail

TOKEN="${HYDRA_TOKEN:-local-development-token-32-bytes}"
API="${HYDRA_API:-http://127.0.0.1:8443/v1/graphs/default/query}"
ADMIN="${HYDRA_ADMIN:-http://127.0.0.1:9090}"

pass=0; fail=0

q() { # q <label> <cypher> [expect-substring]
  local label="$1" cypher="$2" expect="${3:-}"
  local body resp
  body=$(python3 -c 'import json,sys; print(json.dumps({"cell_id":"cell-0","query":sys.argv[1]}))' "$cypher")
  resp=$(curl -sS "$API" -H "Authorization: Bearer $TOKEN" \
         -H 'X-Graph-Namespace: default' -H 'Content-Type: application/json' \
         --data "$body" 2>&1)

  if grep -q '"error"' <<<"$resp"; then
    printf '  FAIL  %-42s %s\n' "$label" \
      "$(python3 -c 'import sys,json; print(json.load(sys.stdin)["error"]["message"])' <<<"$resp" 2>/dev/null || echo "$resp")"
    ((fail++)); return 1
  fi
  if [[ -n "$expect" ]] && ! grep -q "$expect" <<<"$resp"; then
    printf '  FAIL  %-42s expected %q in response\n' "$label" "$expect"
    ((fail++)); return 1
  fi
  printf '  ok    %-42s\n' "$label"
  ((pass++)); return 0
}

echo "== admin =="
if [[ "$(curl -s -o /dev/null -w '%{http_code}' "$ADMIN/readyz")" == "200" ]]; then
  printf '  ok    %-42s\n' "/readyz"; ((pass++))
else
  printf '  FAIL  %-42s not ready\n' "/readyz"; ((fail++))
fi
if curl -s "$ADMIN/metrics" | grep -q '^graph_runtime_ready 1'; then
  printf '  ok    %-42s\n' "graph_runtime_ready=1"; ((pass++))
else
  printf '  FAIL  %-42s\n' "graph_runtime_ready"; ((fail++))
fi

echo "== write =="
q "one-hop CREATE with properties" \
  'CREATE (a:SmokePkg {id: 9000, name: "smoke-app"})-[:DEPENDS_ON]->(b:SmokePkg {id: 9001, name: "smoke-lib"})'
q "chain 9001 -> 9002" 'CREATE (a {id: 9001})-[:DEPENDS_ON]->(b {id: 9002})'
q "chain 9002 -> 9003" 'CREATE (a {id: 9002})-[:DEPENDS_ON]->(b {id: 9003})'
q "reverse edges for blast radius" 'CREATE (a {id: 9003})-[:USED_BY]->(b {id: 9002})'
q "reverse edges (cont.)"          'CREATE (a {id: 9002})-[:USED_BY]->(b {id: 9001})'
q "reverse edges (cont.)"          'CREATE (a {id: 9001})-[:USED_BY]->(b {id: 9000})'

echo "== read =="
q "MATCH by id, RETURN property"      'MATCH (a {id: 9000}) RETURN a.name AS name' 'smoke-app'
q "MATCH by string property"          'MATCH (a {name: "smoke-lib"}) RETURN a.id AS id' '9001'
q "variable-length from fixed source" 'MATCH (a {id: 9000})-[:DEPENDS_ON*1..3]->(x) RETURN x.id AS id' '9003'
q "blast radius via reverse edges"    'MATCH (v {id: 9003})-[:USED_BY*1..5]->(x) RETURN x.id AS blast' '9000'

echo "== native path procedures =="
q "algo.SPpaths" \
  'CALL algo.SPpaths({sourceNode: 9000, targetNode: 9003, relTypes: ["DEPENDS_ON"]}) YIELD path RETURN path' \
  '"path"'
q "algo.SSpaths" \
  'CALL algo.SSpaths({sourceNode: 9000, relTypes: ["DEPENDS_ON"]}) YIELD path RETURN path' \
  '"path"'
q "algo.MSpaths" \
  'CALL algo.MSpaths({sourceLabel: "SmokePkg", sourceProperty: "name", sourceValues: ["smoke-app"], targetLabel: "SmokePkg", targetProperty: "name", targetValues: ["smoke-lib"], relTypes: ["DEPENDS_ON"]}) YIELD path RETURN path' \
  '"path"'

echo
echo "  $pass passed, $fail failed"
[[ $fail -eq 0 ]] || exit 1
