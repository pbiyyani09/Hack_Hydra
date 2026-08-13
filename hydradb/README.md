# Local HydraDB

Single-node HydraDB for development, backed by the local filesystem instead of
S3 so no cloud credentials are needed.

## Run it

```bash
./hydradb/up.sh              # start (writes .env, then docker compose up -d)
./hydradb/smoke-test.sh      # verify it works
docker compose -f hydradb/docker-compose.yml logs -f
docker compose -f hydradb/docker-compose.yml down
```

Data persists in `hydradb/data/` (gitignored). Delete that directory for a
clean slate.

## Endpoints

| Port | Protocol | Use |
|------|----------|-----|
| 7687 | Bolt | Neo4j-compatible drivers (`neo4j://127.0.0.1:7687`) |
| 8443 | HTTP | JSON / NDJSON query API |
| 9090 | Admin | `/readyz`, `/livez`, `/metrics` |

Health is at **`/readyz`** and **`/livez`** — there is no `/health`.
`curl -s localhost:9090/metrics | grep graph_runtime_ready` should report `1`.

Local auth token: `local-development-token-32-bytes` (in `data/auth-token`).

Over HTTP — note the required `cell_id` and `X-Graph-Namespace`:

```bash
curl -sS http://127.0.0.1:8443/v1/graphs/default/query \
  -H "Authorization: Bearer local-development-token-32-bytes" \
  -H 'X-Graph-Namespace: default' \
  -H 'Content-Type: application/json' \
  --data '{"cell_id":"cell-0","query":"MATCH (a {id: 1})-[:FOLLOWS]->(b) RETURN b.id AS id"}'
```

Over Bolt, with the stock Neo4j driver (`pip install neo4j`). Authenticate with
**`bearer_auth`**, not a username/password pair, and pass `database="default"`:

```python
from neo4j import GraphDatabase, bearer_auth

drv = GraphDatabase.driver("neo4j://127.0.0.1:7687",
                           auth=bearer_auth("local-development-token-32-bytes"))
with drv.session(database="default") as s:
    for r in s.run("MATCH (a {id: 9000})-[:DEPENDS_ON*1..3]->(x) RETURN x.id AS id"):
        print(r["id"])
```

---

## Engine capabilities, v0.1.0

Verified by probing this build directly. **The Cypher surface is a narrow
subset**, and these limits shape the data model rather than the other way
around — read this before designing anything.

### Supported

| Capability | Example |
|---|---|
| One-hop `CREATE` with an edge | `CREATE (a {id: 1})-[:R]->(b {id: 2})` |
| Properties inside a one-hop `CREATE` | `CREATE (a {id: 30, name: "left-pad"})-[:R]->(b {id: 31})` |
| Labels on nodes | `CREATE (a:Pkg {id: 10})-[:R]->(b:Pkg {id: 11})` |
| `MATCH` anchored by id, label, or property | `MATCH (a {name: "left-pad"}) RETURN a.id AS id` |
| Multi-hop with **bound** intermediates | `MATCH (a {id: 100})-[:R]->(m)-[:R]->(x) RETURN x.id AS x` |
| Variable-length **from a fixed source** | `MATCH (a {id: 100})-[:R*1..3]->(x) RETURN x.id AS id` |
| Native path procedures | see below |

### Not supported

| Rejected | Engine message |
|---|---|
| Standalone node `CREATE` | `only one-hop edge patterns are executable in Query engine CREATE` |
| `count()` and aggregates | `property values support integer, float, boolean, and string literals` |
| Bare `MATCH (n)` | `node-only MATCH requires an id, label, or property predicate` |
| **Reverse** variable-length traversal | `variable-length MATCH requires a fixed source id` |
| `UNWIND` over the query transport | `query transport cannot authorize an unsupported Cypher clause` |
| List parameters via `parameters` | `composite parameter $x is only supported as an UNWIND input` |

> **The constraint that drives the data model:** variable-length traversal
> requires a *fixed source id*, so `MATCH (x)-[:DEPENDS_ON*1..n]->(victim)` —
> the reverse-dependency closure, i.e. the exact blast-radius question — is
> **rejected**. Write each dependency edge in **both directions** at ingest
> (`DEPENDS_ON` forward, `USED_BY` reverse) and traverse forward from the
> compromised package instead. Verified working:
>
> ```cypher
> MATCH (v {id: 103})-[:USED_BY*1..5]->(x) RETURN x.id AS blast_radius
> ```
>
> Storage is cheap and object-backed; the doubled edge count is the right
> trade for making the primary query expressible at all.

Because list parameters only bind through `UNWIND`, and `UNWIND` is refused by
the query transport, **all list arguments must be inlined as literals** in the
query string. Escape carefully when building queries programmatically.

### Native path procedures

All three take an inline map and `YIELD path`. The returned `path` carries full
`nodes` and `relationships` objects, so one call replaces a client-side fan-out.

```cypher
-- single source -> single target, both by vertex id
CALL algo.SPpaths({sourceNode: 100, targetNode: 103, relTypes: ["DEPENDS_ON"]})
YIELD path RETURN path

-- single source -> every reachable target, by vertex id
CALL algo.SSpaths({sourceNode: 100, relTypes: ["DEPENDS_ON"]})
YIELD path RETURN path

-- many -> many, resolved by property lookup rather than raw ids
CALL algo.MSpaths({
  sourceLabel: "Pkg", sourceProperty: "name", sourceValues: ["app-a"],
  targetLabel: "Pkg", targetProperty: "name", targetValues: ["lib-b"],
  relTypes: ["DEPENDS_ON"]
}) YIELD path RETURN path
```

`relTypes` must be a non-empty list. `SSpaths` accepts neither `targetNodes`
nor `maxHops`. `MSpaths` requires all six keys — omitting any one produces a
`missing OpenCypher query parameter $<key>` error, which is the fastest way to
discover a signature.

## Notes

- `RUST_MIN_STACK=33554432` is required; Cypher planning recurses deeply enough
  to overflow the default 8 MB stack.
- `GRAPH_ALLOW_PLAINTEXT=true` is for local development only.
- HydraDB itself is AGPL-3.0. We run it as a separate container and connect as
  a client over Bolt/HTTP, so our own code is not a derivative work.
