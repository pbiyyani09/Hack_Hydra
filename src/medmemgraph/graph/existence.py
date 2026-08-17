"""existence.py — the ONLY sanctioned existence check (decisions/003, ARCHITECTURE.md §7.6).

Decision 003, executed against `ghcr.io/hydra-db/hydradb:0.1.1`: an
**unlabelled** node pattern matched by `id` (`MATCH (n {id: X}) RETURN n.id
AS id`) returns a row for ANY integer, whether or not the node exists —
`count(*)` inherits the phantom, and a `WHERE` guard does not filter it.
A label predicate fixes it (`MATCH (n:Patient {id: X}) ...`).

`structural_absence` — the project's headline abstention signal, covering
5,964 adversarial items (33.3% of MedLoCoMo) — is only correct if every
existence probe in the codebase goes through this function (or an
edge-anchored / `algo.*paths` shape with the same labelled discipline).
Do not hand-roll `MATCH (n {id: ...})` anywhere else.
"""

from __future__ import annotations

import re

from medmemgraph.hydra_client import HydraClient

_LABEL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def exists(label: str, node_id: int, *, client: HydraClient | None = None) -> bool:
    """True iff a node with this label and integer `id` exists.

    ALWAYS takes a label — there is no unlabelled overload. HydraDB has no
    `IS NULL` and no reliable `count(*)` on an unlabelled pattern
    (decisions/003), so "does this id exist" is answered by row count on a
    labelled `MATCH`, never by an unlabelled lookup.

    `client`: reuse an existing `HydraClient` (e.g. in a test fixture or a
    hot retrieval path) to avoid opening a new Bolt connection per call. If
    omitted, a short-lived client is opened and closed around this one call.
    """
    if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
        raise ValueError(
            f"exists(): label must be a non-empty Cypher identifier, got {label!r} "
            "— refusing to build an unlabelled or unsafe MATCH pattern (decisions/003)."
        )

    owns_client = client is None
    active_client = client or HydraClient(transport="bolt")
    try:
        rows = active_client.run(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.id AS id",
            id=node_id,
        )
        return len(rows) > 0
    finally:
        if owns_client:
            active_client.close()
