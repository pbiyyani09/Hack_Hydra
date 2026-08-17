"""tests/test_probe.py — GT-first contradiction/update/correction probe
(pipeline/probe.py). Written alongside the story (no separate test-lead
artifact exists in this repo's coordination model, same convention every
prior `[dev-ml]` entry in `.claude/logs/dev.log.md` has noted).

Runs fully offline against a small synthetic MedLoCoMo-shaped corpus built
directly in `tmp_path` (never the real corpus, never `ANTHROPIC_API_KEY`) —
same discipline as `tests/test_resolve.py` / `tests/test_reader.py`. The
QC gates under test (`_run_qc`) are themselves deterministic (no live model
call — see `probe.py`'s module docstring), so this suite never skips for a
missing API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest

from medmemgraph.contracts import PREDICATES, VALID_POLARITIES
from medmemgraph.pipeline import probe
from medmemgraph.pipeline.loader import load_qa

# ---------------------------------------------------------------------------
# Synthetic corpus builder — mirrors combined_conversation.json's real shape
# (loader._parse_conversation's required keys) so probe.py exercises the
# same allowlisted loader path a real run would.
# ---------------------------------------------------------------------------


def _write_patient(root: Path, subject_id: str, admissions: list[tuple[str, str, str, int]]) -> None:
    """`admissions`: list of (hadm_id, admission_start, admission_end, n_turns)."""
    doc = {
        "subject_id": subject_id,
        "processed_hadm_ids": [hadm_id for hadm_id, *_ in admissions],
        "admissions": [],
    }
    for hadm_id, start, end, n_turns in admissions:
        lines = []
        base_hour = 8
        for i in range(1, n_turns + 1):
            speaker = "Doctor" if i % 2 == 1 else "Patient"
            lines.append(
                {
                    "turn_number": i,
                    "time": f"{start[:10]} {base_hour:02d}:{i:02d}:00",
                    "speaker": speaker,
                    "text": f"[{hadm_id} turn {i}] routine filler dialogue about recovery.",
                }
            )
        doc["admissions"].append(
            {
                "hadm_id": hadm_id,
                "admission_start": start,
                "admission_end": end,
                "conversation_lines": lines,
            }
        )
    patient_dir = root / "MedLoCoMo" / subject_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    (patient_dir / "combined_conversation.json").write_text(json.dumps(doc))


@pytest.fixture()
def small_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write_patient(
        root,
        "P0001",
        [
            ("H1", "2124-01-01 08:00:00", "2124-01-03 12:00:00", 6),
            ("H2", "2124-03-01 08:00:00", "2124-03-03 12:00:00", 6),
            ("H3", "2124-06-01 08:00:00", "2124-06-03 12:00:00", 6),
        ],
    )
    _write_patient(
        root,
        "P0002",
        [
            ("H1", "2124-02-01 08:00:00", "2124-02-04 12:00:00", 5),
            ("H2", "2124-05-01 08:00:00", "2124-05-04 12:00:00", 5),
        ],
    )
    _write_patient(
        root,
        "P0003",
        [("H1", "2124-01-01 08:00:00", "2124-01-02 12:00:00", 4)],  # only 1 admission -- must be skipped
    )
    return root


# ---------------------------------------------------------------------------
# 1. Ground truth is fixed BEFORE generation — the structural ordering.
# ---------------------------------------------------------------------------


class TestGroundTruthFirst:
    def test_scenario_builders_need_no_corpus(self) -> None:
        """Every builder takes only a Random — no Conversation/Admission
        object exists anywhere in scope when the ground truth is built.
        This is the structural proof, not an assertion: it would be
        impossible to write this test if a builder's signature required
        dialogue text."""
        rng = Random(0)
        for builder in probe._SCENARIO_BUILDERS:
            spec = builder(rng)  # noqa: no Conversation/Admission constructed above this line
            assert spec.kind in probe.PROBE_KINDS
            assert spec.predicate in PREDICATES
            assert spec.old_polarity in VALID_POLARITIES
            assert spec.new_polarity in VALID_POLARITIES
            assert spec.expected_answer_now
            assert spec.expected_answer_as_of
            assert spec.old_fragment.lower() in spec.old_quote.lower()
            assert spec.new_fragment.lower() in spec.new_quote.lower()

    def test_ground_truth_fields_unchanged_after_dialogue_injection(self, small_corpus: Path) -> None:
        """The GT fields on the returned ProbeItem must equal exactly what
        the scenario spec said before any turn was ever appended — nothing
        downstream is allowed to "fix up" the answer after looking at the
        generated text (the LoCoMo failure mode, literature/08 R-SYN-15)."""
        rng = Random(42)
        spec = probe._scenario_dose_change(rng)
        conv = probe.load_conversation("P0001", str(small_corpus))

        item, reason = probe._try_build_item(conv, spec, Random(1), probe_index=0, used_signatures=set())
        assert item is not None, reason

        assert item.original_fact.value_text == spec.old_value_text
        assert item.superseding_fact.value_text == spec.new_value_text
        assert item.expected_answer_now == spec.expected_answer_now
        assert item.expected_answer_as_of == spec.expected_answer_as_of
        assert item.original_fact.polarity == spec.old_polarity
        assert item.superseding_fact.polarity == spec.new_polarity
        assert item.predicate == spec.predicate


# ---------------------------------------------------------------------------
# 2. Injected turns land in the LATER admission.
# ---------------------------------------------------------------------------


class TestInjectionLocation:
    def test_superseding_fact_in_later_admission(self, small_corpus: Path) -> None:
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(7))
        item, reason = probe._try_build_item(conv, spec, Random(7), probe_index=0, used_signatures=set())
        assert item is not None, reason

        admissions_by_id = {a.hadm_id: a for a in conv.admissions}
        old_admission = admissions_by_id[item.original_fact.session_id]
        new_admission = admissions_by_id[item.superseding_fact.session_id]
        assert old_admission.admission_start < new_admission.admission_start, (
            "original_fact must live in the chronologically EARLIER admission"
        )

    def test_injected_lines_actually_present_in_the_copy(self, small_corpus: Path) -> None:
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(3))
        item, reason = probe._try_build_item(conv, spec, Random(3), probe_index=0, used_signatures=set())
        assert item is not None, reason

        injected_by_id = {a.hadm_id: a for a in item.injected_conversation.admissions}
        old_lines = injected_by_id[item.original_fact.session_id].conversation_lines
        new_lines = injected_by_id[item.superseding_fact.session_id].conversation_lines

        assert any(
            line["turn_number"] == item.original_fact.turn_ids[0] and line["text"] == item.original_fact.quote
            for line in old_lines
        )
        assert any(
            line["turn_number"] == item.superseding_fact.turn_ids[0] and line["text"] == item.superseding_fact.quote
            for line in new_lines
        )

    def test_corpus_on_disk_never_mutated(self, small_corpus: Path) -> None:
        """The real on-disk fixture must be byte-identical after build_probe
        runs — injection only ever touches the in-memory copy."""
        path = small_corpus / "MedLoCoMo" / "P0001" / "combined_conversation.json"
        before = path.read_bytes()
        probe.build_probe("P0001", n_patients=3, seed=1, root=str(small_corpus))
        after = path.read_bytes()
        assert before == after


# ---------------------------------------------------------------------------
# 3. The three case types are distinguishable.
# ---------------------------------------------------------------------------


class TestThreeKinds:
    def test_all_three_kinds_are_reachable(self) -> None:
        rng = Random(0)
        kinds = {builder(rng).kind for builder in probe._SCENARIO_BUILDERS}
        assert kinds == set(probe.PROBE_KINDS) == {"update", "correction", "contradiction"}

    def test_contradiction_answer_flags_conflict_update_and_correction_do_not(self) -> None:
        rng = Random(0)
        by_kind: dict[str, list] = {"update": [], "correction": [], "contradiction": []}
        for builder in probe._SCENARIO_BUILDERS:
            spec = builder(rng)
            by_kind[spec.kind].append(spec)

        for spec in by_kind["contradiction"]:
            assert "conflicting" in spec.expected_answer_now.lower()
        for spec in by_kind["update"] + by_kind["correction"]:
            assert "conflicting" not in spec.expected_answer_now.lower()

        # Update and correction are mechanically identical (asserted -> negated
        # on the same object key) but semantically distinct: only correction's
        # injected superseding quote frames the earlier statement as wrong.
        for spec in by_kind["correction"]:
            assert "error" in spec.new_quote.lower()
        for spec in by_kind["update"]:
            assert spec.new_polarity in VALID_POLARITIES  # sanity: still a legal polarity


# ---------------------------------------------------------------------------
# 4. QC gates actually reject a deliberately-bad item.
# ---------------------------------------------------------------------------


class TestQCGatesReject:
    def test_evidence_ablation_gate_rejects_a_leaked_fragment(self, small_corpus: Path) -> None:
        """Deliberately construct a scenario whose grounding fragment is
        ALREADY present in the patient's real (un-injected) dialogue —
        the gate must catch it."""
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(0))
        # Sabotage: rewrite the quote/fragment pair so the fragment stays a
        # legal substring of the quote (passes _validate_spec) but is a
        # phrase that already occurs in the real filler text every
        # synthetic admission carries ("routine filler dialogue about
        # recovery" — see `_write_patient`).
        spec.old_quote = "This turn mentions routine filler dialogue about recovery, for testing."
        spec.old_fragment = "routine filler dialogue about recovery"

        item, reason = probe._try_build_item(conv, spec, Random(0), probe_index=0, used_signatures=set())
        # The collision guard inside _try_build_item itself should already
        # refuse to build this item (defense layer 1).
        assert item is None
        assert "collides" in reason

    def test_run_qc_directly_flags_ablation_failure(self, small_corpus: Path) -> None:
        """Bypass the construction-time guard entirely and call `_run_qc`
        on a hand-corrupted item, to prove the QC function itself (not just
        the earlier guard) detects the leak — defense layer 2."""
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(5))
        item, reason = probe._try_build_item(conv, spec, Random(5), probe_index=0, used_signatures=set())
        assert item is not None, reason

        # Corrupt the item after honest construction: claim its key fragment
        # is something that is already present in the real corpus text.
        corrupted = probe.replace(
            item.original_fact, key_fragment="routine filler dialogue"
        )
        item.original_fact = corrupted

        qc = probe._run_qc(item, conv)
        assert qc.evidence_ablation_ok is False
        assert qc.passed is False
        assert "evidence_ablation" in qc.failing_gates()

    def test_run_qc_flags_recency_leak(self, small_corpus: Path) -> None:
        """Corrupt the *injected* conversation so the old fragment also
        appears in the most recent admission — the last-session-only gate
        must catch it even though the ablation gate (which checks the
        un-injected `conv`) would not."""
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(9))
        item, reason = probe._try_build_item(conv, spec, Random(9), probe_index=0, used_signatures=set())
        assert item is not None, reason

        admissions = list(item.injected_conversation.admissions)
        recency_admission = max(admissions, key=lambda a: a.admission_start)
        leaked_line = {
            "turn_number": 999,
            "time": recency_admission.admission_end,
            "speaker": "Doctor",
            "text": item.original_fact.quote,  # leak the OLD quote into the newest admission
        }
        leaked_admission = probe.replace(
            recency_admission, conversation_lines=recency_admission.conversation_lines + (leaked_line,)
        )
        new_admissions = tuple(
            leaked_admission if a.hadm_id == recency_admission.hadm_id else a for a in admissions
        )
        item.injected_conversation = probe.replace(item.injected_conversation, admissions=new_admissions)

        qc = probe._run_qc(item, conv)
        assert qc.last_session_only_ok is False
        assert "last_session_only" in qc.failing_gates()

    def test_run_qc_passes_a_clean_item(self, small_corpus: Path) -> None:
        conv = probe.load_conversation("P0001", str(small_corpus))
        spec = probe._scenario_dose_change(Random(11))
        item, reason = probe._try_build_item(conv, spec, Random(11), probe_index=0, used_signatures=set())
        assert item is not None, reason
        qc = probe._run_qc(item, conv)
        assert qc.passed is True, qc.to_dict()


# ---------------------------------------------------------------------------
# build_probe end-to-end + drop reporting
# ---------------------------------------------------------------------------


class TestBuildProbeEndToEnd:
    def test_build_probe_over_synthetic_corpus(self, small_corpus: Path) -> None:
        items, report = probe.build_probe(
            None, n_patients=2, seed=3, root=str(small_corpus), return_report=True
        )
        assert 1 <= len(items) <= 2  # P0003 (1 admission) is structurally ineligible
        for item in items:
            assert item.qc is not None and item.qc.passed
            assert item.predicate in PREDICATES
            assert item.kind in probe.PROBE_KINDS
            assert item.patient_id in ("P0001", "P0002")
        assert report["n_generated"] == len(items)
        assert report["n_requested"] == 2

    def test_single_patient_mode(self, small_corpus: Path) -> None:
        items = probe.build_probe("P0001", n_patients=3, seed=5, root=str(small_corpus))
        assert len(items) <= 3
        assert all(item.patient_id == "P0001" for item in items)

    def test_declared_stopping_point_returns_fewer_not_infinite_loop(self, small_corpus: Path) -> None:
        """Requesting more patients than the corpus has eligible (>=2
        admissions) patients must terminate promptly and return at most the
        number of real eligible patients — never hang. P0003 (only 1
        admission) is filtered out during candidate selection and must
        never appear as a carrier at all."""
        items, report = probe.build_probe(
            None, n_patients=10, seed=0, root=str(small_corpus), return_report=True
        )
        assert len(items) <= 2  # only P0001 and P0002 have >= 2 admissions
        assert report["n_requested"] <= 2
        assert all(item.patient_id != "P0003" for item in items)


# ---------------------------------------------------------------------------
# 5. export() matches the benchmark_qa.json schema.
# ---------------------------------------------------------------------------


class TestExportSchema:
    def test_export_shape_and_round_trip_through_load_qa(self, small_corpus: Path, tmp_path: Path) -> None:
        items = probe.build_probe(None, n_patients=2, seed=2, root=str(small_corpus))
        assert items, "fixture must yield at least one QC-passing item"

        out_path = probe.export(items, tmp_path / "probe_bundle" / "probe_qa.json")
        assert out_path.exists()

        raw = json.loads(out_path.read_text())
        assert set(raw.keys()) == {"qas"}
        assert len(raw["qas"]) == 2 * len(items)

        for qa in raw["qas"]:
            assert set(qa.keys()) == {"qa_id", "scope", "question_type", "question", "answer", "evidence"}
            assert qa["scope"] in ("single_admission", "cross_admission")
            assert qa["question_type"] in probe.PROBE_KINDS
            assert isinstance(qa["question"], str) and qa["question"]
            assert isinstance(qa["answer"], str) and qa["answer"]
            assert isinstance(qa["evidence"]["admissions"], list)
            assert all(isinstance(a, str) for a in qa["evidence"]["admissions"])
            if "turn_ids" in qa["evidence"]:
                assert all(isinstance(t, int) for t in qa["evidence"]["turn_ids"])

        # Round-trip through the REAL production loader — proves this is not
        # just eyeballed-similar JSON but something load_qa actually accepts.
        fake_root = tmp_path / "as_benchmark_root"
        fake_patient_dir = fake_root / "MedLoCoMo" / "PROBE_EXPORT_CHECK"
        fake_patient_dir.mkdir(parents=True)
        (fake_patient_dir / "benchmark_qa.json").write_text(json.dumps(raw))
        loaded = load_qa("PROBE_EXPORT_CHECK", str(fake_root))
        assert loaded == raw["qas"]

    def test_qc_summary_reports_gate_level_counts(self, small_corpus: Path) -> None:
        items, report = probe.build_probe(
            None, n_patients=2, seed=2, root=str(small_corpus), return_report=True
        )
        text = probe.qc_summary(report)
        assert "requested=" in text
        assert "generated=" in text
        assert "drop_rate=" in text
