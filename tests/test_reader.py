"""tests/test_reader.py — coverage for the Chain-of-Note extract-then-reason
reader (`medmemgraph.eval.reader`).

Everything here runs offline: the `dry_run=True` path exercises the
deterministic stub (no API key needed, matches the rest of this project's
test suite), and a handful of tests replace `medmemgraph.llm`'s lazy
OpenAI-client singleton with an in-process fake object via
`monkeypatch.setattr(llm, "_get_openai_client", ...)` to exercise the
real-call JSON-parsing / token-accounting code path (`reader.py` ->
`llm.complete()` -> the OpenAI provider branch) without a live network
call — the same pattern `tests/test_llm.py` establishes for testing
consumers of the seam. The autouse `isolate_llm_module` fixture points
`llm.CACHE_DIR` at a per-test `tmp_path` and resets the seam's mutable
singletons, so this file never touches the real repo `data/llm_cache/` or
the real `.env` (which, in this repo, really does carry a live
`OPEN_AI_KEY` / `GOOGLE_API_KEY` — see `TestMissingKeyRaisesRatherThanDegrading`
below, which specifically needs that NOT to leak in).
"""

from __future__ import annotations

import inspect
import json
import random

import pytest

from medmemgraph import llm
from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.eval.harness import evaluate
from medmemgraph.eval.reader import (
    Answer,
    Note,
    ReaderChainOfNoteAnswerer,
    ReaderDirectAnswerer,
    _default_retriever,
    _split_timestamp,
    read,
    render_context,
)
from medmemgraph.eval.types import estimate_tokens
from medmemgraph.pipeline.loader import Admission, Conversation


# ---------------------------------------------------------------------------
# Isolation fixture — fresh cache dir, ledger, and client singletons; the
# real repo .env / data/llm_cache/ are never touched by this file.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_llm_module(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_ledger", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_google_client", None)
    monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    yield


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _item(text: str, session_id: str = "H1", turn_ids=None, score: float = 0.9, channel: str = "vector") -> RetrieveItem:
    return RetrieveItem(text=text, session_id=session_id, turn_ids=turn_ids or [1], score=score, channel=channel)


# ---------------------------------------------------------------------------
# Fakes — OpenAI-shaped, matching the attribute shapes llm.py reads off a
# real `openai` SDK response (mirrors tests/test_llm.py's own fakes,
# duplicated here rather than imported, per this project's "each test file
# fully self-contained" convention — see eval/baselines/dense.py's
# `_TimedRetriever` docstring for the same reasoning stated once). Reader
# calls use `llm.ANSWER_MODEL` (OpenAI, gpt-4.1-mini) by default, so the
# fake models the OpenAI response shape, not the Google one `test_judge.py`
# uses for the (Google) judge.
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeOpenAIChat:
    def __init__(self, payload: dict, input_tokens: int, output_tokens: int) -> None:
        self._payload = payload
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeChatCompletion(json.dumps(self._payload), self.input_tokens, self.output_tokens)


class FakeClient:
    """Stand-in for `openai.OpenAI()`, scripted to return a fixed JSON
    payload. Installed via `monkeypatch.setattr(llm, "_get_openai_client",
    lambda: client)` — never passed to `read()` directly (the old
    `client=` kwarg on `read()` was removed along with the direct-Anthropic
    call path it existed to support; `llm.py` owns client construction now,
    see `reader.py`'s module docstring)."""

    def __init__(self, payload: dict, *, input_tokens: int = 111, output_tokens: int = 22):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeOpenAIChat(payload, input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# 1. Both modes produce a valid Answer (offline stub)
# ---------------------------------------------------------------------------


class TestBothModesProduceAValidAnswer:
    @pytest.mark.parametrize("mode", ["direct", "chain_of_note"])
    def test_read_returns_answer_dataclass(self, mode):
        items = [_item("Patient takes metformin 500mg for diabetes management.")]
        answer = read("What medication does the patient take?", items, mode, dry_run=True)
        assert isinstance(answer, Answer)
        assert isinstance(answer.text, str) and answer.text
        assert isinstance(answer.abstained, bool)
        assert isinstance(answer.notes, list)
        assert answer.mode == mode
        assert answer.prompt_tokens > 0
        assert answer.completion_tokens > 0
        assert answer.latency_ms >= 0.0
        assert answer.total_tokens == answer.prompt_tokens + answer.completion_tokens

    def test_direct_mode_never_produces_notes(self):
        items = [_item("Patient takes metformin 500mg.")]
        answer = read("dose?", items, "direct", dry_run=True)
        assert answer.notes == []
        assert answer.citations == []

    def test_chain_of_note_mode_produces_one_note_per_item(self):
        items = [
            _item("Patient takes metformin 500mg.", session_id="H1", turn_ids=[4]),
            _item("Patient reports chest pain.", session_id="H2", turn_ids=[7]),
        ]
        answer = read("what dose of metformin?", items, "chain_of_note", dry_run=True)
        assert len(answer.notes) == len(items)
        assert all(isinstance(n, Note) for n in answer.notes)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown reader mode"):
            read("q", [_item("x")], "creative_writing", dry_run=True)

    def test_invalid_rendering_raises(self):
        with pytest.raises(ValueError, match="unknown rendering strategy"):
            read("q", [_item("x")], "direct", rendering="xml", dry_run=True)


# ---------------------------------------------------------------------------
# 2. Abstention is a structured field, not a regex over prose
# ---------------------------------------------------------------------------


class TestAbstentionIsAStructuredField:
    def test_empty_pack_abstains_with_no_model_call(self):
        answer = read("anything?", [], "chain_of_note", dry_run=True)
        assert answer.abstained is True
        assert answer.text == "NOT_IN_RECORD"
        assert answer.notes == []
        assert answer.prompt_tokens == 0 and answer.completion_tokens == 0

    def test_structural_absence_forces_abstention_in_stub(self):
        items = [_item("Some unrelated note.")]
        answer = read(
            "Is the patient allergic to anything?", items, "chain_of_note",
            structural_absence=True, dry_run=True,
        )
        assert answer.abstained is True
        assert all(not n.relevant for n in answer.notes)

    def test_abstained_field_is_trusted_even_when_prose_has_no_refusal_keywords(self, monkeypatch):
        """The scripted completion sets abstained=True while the answer text
        itself contains no refusal-shaped words at all -- proving `abstained`
        is read from the structured JSON field, not inferred by matching a
        phrase in `.text` (the story's explicit "not a parse failure"
        requirement)."""
        payload = {"answer": "Purple elephants dance on Tuesdays.", "abstained": True}
        client = FakeClient(payload)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        answer = read(
            "What is the patient's blood type?", [_item("irrelevant evidence")], "direct",
            dry_run=False,
        )
        assert answer.abstained is True
        assert answer.text == "Purple elephants dance on Tuesdays."
        assert "not" not in answer.text.lower() and "record" not in answer.text.lower()

    def test_answer_text_not_in_record_also_counts_as_abstained_even_if_flag_missing(self, monkeypatch):
        """Defense in depth: if a scripted/real completion forgets the
        boolean but still emits the literal NOT_IN_RECORD sentinel required
        by the schema's own instructions, abstention is still detected."""
        payload = {"answer": "NOT_IN_RECORD", "abstained": False}
        client = FakeClient(payload)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        answer = read("q", [_item("x")], "direct", dry_run=False)
        assert answer.abstained is True


# ---------------------------------------------------------------------------
# 3. All three rendering strategies produce parseable context
# ---------------------------------------------------------------------------


class TestRenderingStrategies:
    @pytest.fixture
    def items(self) -> list[RetrieveItem]:
        return [
            _item("[2126-01-05 09:00:00] Patient reports chest pain.", session_id="H1", turn_ids=[1]),
            _item("Patient started on metformin 500mg.", session_id="H1", turn_ids=[2]),
            _item("Follow-up: pain resolved after discharge.", session_id="H2", turn_ids=[3]),
        ]

    def test_json_rendering_round_trips_and_carries_provenance(self, items):
        text = render_context(items, "json")
        payload = json.loads(text)  # must not raise -- "parseable"
        assert len(payload) == len(items)
        for entry, item in zip(payload, items):
            assert entry["session_id"] == item.session_id
            assert entry["turn_ids"] == list(item.turn_ids)
            assert entry["channel"] == item.channel
        # The leading bracketed timestamp on item[0] is split out, not left
        # buried mid-string.
        assert payload[0]["time"] == "2126-01-05 09:00:00"
        assert "2126-01-05" not in payload[0]["text"]
        # Items with no detectable timestamp report it honestly as unknown,
        # never invented.
        assert payload[1]["time"] is None

    def test_prose_rendering_carries_provenance_for_every_item(self, items):
        text = render_context(items, "prose")
        assert isinstance(text, str) and text
        for item in items:
            assert item.session_id in text
            assert str(item.turn_ids) in text or str(list(item.turn_ids)) in text
        # Natural order preserved (no shuffle requested).
        assert text.index("Item 1") < text.index("Item 2") < text.index("Item 3")

    def test_shuffled_rendering_is_parseable_and_reorders_display(self, items):
        # Use a longer list so an identity permutation is astronomically
        # unlikely for a fixed seed (8! = 40320 possibilities).
        many_items = [_item(f"finding number {i}", session_id=f"H{i}", turn_ids=[i]) for i in range(8)]
        text = render_context(many_items, "shuffled")
        assert isinstance(text, str) and text
        for item in many_items:
            assert item.session_id in text
        item_order = [int(n) for n in __import__("re").findall(r"Item (\d+)", text)]
        assert sorted(item_order) == list(range(1, 9))  # every item present exactly once
        assert item_order != list(range(1, 9))  # but not left in natural order

    def test_shuffled_rendering_is_deterministic_given_the_same_rng_seed(self, items):
        text_a = render_context(items, "shuffled", rng=random.Random(7))
        text_b = render_context(items, "shuffled", rng=random.Random(7))
        assert text_a == text_b

    @pytest.mark.parametrize("rendering", ["json", "prose", "shuffled"])
    def test_read_accepts_every_rendering_strategy(self, items, rendering):
        answer = read("What did the follow-up find?", items, "chain_of_note", rendering=rendering, dry_run=True)
        assert answer.rendering == rendering
        assert isinstance(answer.text, str)


# ---------------------------------------------------------------------------
# 4. Token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    def test_dry_run_prompt_tokens_grow_with_more_evidence(self):
        one_item = [_item("short note")]
        many_items = [_item(f"a fairly long clinical finding, number {i}, about the patient") for i in range(10)]
        small = read("q", one_item, "chain_of_note", dry_run=True)
        big = read("q", many_items, "chain_of_note", dry_run=True)
        assert big.prompt_tokens > small.prompt_tokens

    def test_real_call_token_counts_come_from_usage_not_estimation(self, monkeypatch):
        """When a real (or fake, scripted) client is used, token counts must
        be the API's own reported usage -- not `estimate_tokens` -- so cost
        reporting matches what was actually billed."""
        payload = {"answer": "metformin 500mg", "abstained": False}
        client = FakeClient(payload, input_tokens=987, output_tokens=13)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        answer = read("dose?", [_item("Patient takes metformin 500mg.")], "direct", dry_run=False)
        assert answer.prompt_tokens == 987
        assert answer.completion_tokens == 13

    def test_real_call_requests_temperature_zero(self, monkeypatch):
        payload = {"answer": "x", "abstained": False}
        client = FakeClient(payload)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        read("q", [_item("x")], "direct", dry_run=False, model="gpt-4.1-mini")
        assert client.chat.completions.last_kwargs["temperature"] == 0.0

    def test_empty_pack_reports_zero_tokens(self):
        answer = read("q", [], "chain_of_note", dry_run=True)
        assert answer.prompt_tokens == 0
        assert answer.completion_tokens == 0
        assert answer.total_tokens == 0


# ---------------------------------------------------------------------------
# 5. "Do not invent turns" — note reconciliation grounds citations in real items
# ---------------------------------------------------------------------------


class TestCitationsAreGroundedNeverInvented:
    def test_metformin_example_answer_mentions_dose_and_cites_the_real_item(self):
        """Mirrors the story's own worked example: one item stating a dose,
        a question asking for that dose -- the answer should surface it and
        the citation should point at the exact (session_id, turn_ids) the
        evidence actually carries."""
        items = [_item("Patient takes metformin 500mg", session_id="H1", turn_ids=[4])]
        answer = read("what dose of metformin?", items, "chain_of_note", dry_run=True)
        assert "500" in answer.text
        assert {"session_id": "H1", "turn_ids": [4]} in answer.citations

    def test_a_note_naming_a_session_not_in_the_pack_is_dropped_not_trusted(self, monkeypatch):
        """Scripted completion invents a citation to a session/turn that was
        never retrieved. `_reconcile_notes` must not surface it as a real
        citation -- the note is grounded from the real item list only."""
        items = [_item("Patient takes metformin 500mg.", session_id="H1", turn_ids=[4])]
        payload = {
            "notes": [
                {"session_id": "H1", "turn_ids": [4], "note": "takes metformin 500mg", "relevant": True},
                # Hallucinated extra note for an item that was never retrieved:
                {"session_id": "H99", "turn_ids": [999], "note": "invented fact", "relevant": True},
            ],
            "answer": "500mg",
            "abstained": False,
        }
        client = FakeClient(payload)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        answer = read("dose?", items, "chain_of_note", dry_run=False)
        assert len(answer.notes) == 1  # exactly one note per real retrieved item, no more
        assert answer.citations == [{"session_id": "H1", "turn_ids": [4]}]
        assert not any(n.session_id == "H99" for n in answer.notes)

    def test_skipped_item_defaults_to_irrelevant_not_silently_dropped(self, monkeypatch):
        """If the model's notes array omits an item entirely, that item
        still gets a (non-citable) Note -- "do not skip any item" is
        enforced structurally, not just requested in the prompt."""
        items = [
            _item("Patient takes metformin 500mg.", session_id="H1", turn_ids=[4]),
            _item("Patient reports headache.", session_id="H2", turn_ids=[9]),
        ]
        payload = {
            "notes": [{"session_id": "H1", "turn_ids": [4], "note": "takes metformin 500mg", "relevant": True}],
            "answer": "500mg",
            "abstained": False,
        }
        client = FakeClient(payload)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: client)
        answer = read("dose?", items, "chain_of_note", dry_run=False)
        assert len(answer.notes) == 2
        skipped = next(n for n in answer.notes if n.session_id == "H2")
        assert skipped.relevant is False
        assert skipped.text == "IRRELEVANT"


# ---------------------------------------------------------------------------
# 5b. Missing key + not dry_run raises, rather than degrading (story's
#     explicit hard requirement: "the stub must become explicit, never
#     silent"). Mirrors tests/test_judge.py's equivalent coverage.
# ---------------------------------------------------------------------------


class TestMissingKeyRaisesRatherThanDegrading:
    def test_no_key_and_dry_run_false_raises_missing_api_key_error(self, monkeypatch):
        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("OpenAI API key not found (simulated for test).")

        monkeypatch.setattr(llm, "resolve_openai_key", _raise_missing)
        with pytest.raises(llm.MissingAPIKeyError):
            read("dose?", [_item("Patient takes metformin 500mg.")], "direct", dry_run=False)

    def test_no_key_and_dry_run_false_raises_for_chain_of_note_mode_too(self, monkeypatch):
        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("OpenAI API key not found (simulated for test).")

        monkeypatch.setattr(llm, "resolve_openai_key", _raise_missing)
        with pytest.raises(llm.MissingAPIKeyError):
            read("dose?", [_item("Patient takes metformin 500mg.")], "chain_of_note", dry_run=False)

    def test_dry_run_true_never_needs_a_key_even_if_resolution_would_fail(self, monkeypatch):
        """The inverse case, to prove the stub path is genuinely
        key-independent: even with key resolution rigged to always fail,
        `dry_run=True` must still succeed (it never calls
        resolve_openai_key at all -- `read()`'s own empty-items early
        return and `_stub_complete` are both pure/offline)."""

        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("should never be called")

        monkeypatch.setattr(llm, "resolve_openai_key", _raise_missing)
        answer = read("dose?", [_item("Patient takes metformin 500mg.")], "direct", dry_run=True)
        assert isinstance(answer, Answer)

    def test_reader_answerer_with_dry_run_false_and_no_key_raises_not_degrades(self, monkeypatch):
        """Same guarantee, exercised through the harness-facing
        `ReaderAnswerer` wrapper rather than `read()` directly."""

        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("OpenAI API key not found (simulated for test).")

        monkeypatch.setattr(llm, "resolve_openai_key", _raise_missing)
        answerer = ReaderChainOfNoteAnswerer(dry_run=False)
        with pytest.raises(llm.MissingAPIKeyError):
            answerer.answer("dose?", None, patient_id="p1")


# ---------------------------------------------------------------------------
# 6. No retrieval inside read() (mirrors the old E7-S5 acceptance criterion:
#    "grepped for retrieve( -> there is no call that performs retrieval")
# ---------------------------------------------------------------------------


class TestReadNeverRetrieves:
    def test_read_source_never_calls_retrieve(self):
        source = inspect.getsource(read)
        assert "retrieve(" not in source
        assert "mock_retrieve(" not in source
        assert "load_conversation(" not in source
        assert "load_qa(" not in source


# ---------------------------------------------------------------------------
# 7. Harness wiring — reader_direct / reader_con are real, selectable systems
# ---------------------------------------------------------------------------


class TestHarnessWiring:
    @pytest.fixture
    def small_conversation(self) -> Conversation:
        admissions = (
            Admission(
                hadm_id="adm1",
                admission_start="2126-01-01",
                admission_end="2126-01-02",
                conversation_lines=(
                    {"turn_number": 1, "time": "2126-01-01T09:00:00", "speaker": "Doctor", "text": "How are you?"},
                ),
            ),
        )
        return Conversation(subject_id="fixture-1", processed_hadm_ids=("adm1",), admissions=admissions)

    @pytest.fixture
    def qa_items(self) -> list[dict]:
        return [
            {
                "qa_id": "q1",
                "scope": "single_admission",
                "question_type": "medical_reasoning",
                "question": "Why did the patient have chest pain?",
                "answer": "unclear etiology",
                "evidence": {"admissions": ["adm1"]},
            }
        ]

    @pytest.mark.parametrize("system_name", ["reader_direct", "reader_con"])
    def test_reader_systems_are_registered_and_runnable_under_dry_run(
        self, system_name, small_conversation, qa_items
    ):
        run = evaluate(
            qa_items, small_conversation, patient_id="fixture-1", system_name=system_name, dry_run=True
        )
        assert run.n_items == 1
        assert run.system_name == system_name
        assert run.records[0].mode == "answerable"

    def test_reader_direct_and_reader_con_are_distinct_classes_selectable_independently(self):
        direct = ReaderDirectAnswerer(dry_run=True)
        con = ReaderChainOfNoteAnswerer(dry_run=True)
        assert direct.name == "reader_direct" and direct.mode == "direct"
        assert con.name == "reader_con" and con.mode == "chain_of_note"

    def test_reader_answerer_ignores_the_conversation_argument(self, small_conversation):
        """The reader system answers over *retrieved* evidence
        (mock_retrieve today), never the raw turn history it's handed for
        Answerer-protocol parity -- mirrors nomem's own contract test."""
        answerer = ReaderChainOfNoteAnswerer(dry_run=True)
        result = answerer.answer("Any question?", small_conversation, patient_id="fixture-1")
        assert result.text  # answered from mock_retrieve's evidence, not raised on `conversation`

    def test_rendering_and_retrieve_k_flow_through_evaluate_without_erroring_other_systems(
        self, small_conversation, qa_items
    ):
        # nomem does not accept rendering/retrieve_k -- must be silently
        # ignored (introspection guard in `_build_answerer`), not raise.
        run = evaluate(
            qa_items, small_conversation, patient_id="fixture-1", system_name="nomem",
            dry_run=True, rendering="prose", retrieve_k=3,
        )
        assert run.n_items == 1
        run2 = evaluate(
            qa_items, small_conversation, patient_id="fixture-1", system_name="reader_con",
            dry_run=True, rendering="prose", retrieve_k=3,
        )
        assert run2.n_items == 1


# ---------------------------------------------------------------------------
# 8. structural_absence flows from RetrieveResult through ReaderAnswerer
# ---------------------------------------------------------------------------


class TestStructuralAbsenceThreadsThroughReaderAnswerer:
    def test_structural_absence_true_forces_abstention_via_retriever(self):
        def fake_retriever(question, patient_id, k):
            return RetrieveResult(
                items=[_item("some evidence")],
                route="graph",
                structural_absence=True,
                paths=[],
                latency_ms={"total": 0.0},
            )

        answerer = ReaderChainOfNoteAnswerer(dry_run=True, retriever=fake_retriever)
        result = answerer.answer("does the patient have X?", None, patient_id="p1")
        assert result.text == "NOT_IN_RECORD"


# ---------------------------------------------------------------------------
# 9. Eval path always pins epsilon=0 (decisions/005 Finding 1). The
# reconciliation audit found `eval/reader.py` had zero references to
# `epsilon` -- `retrieve()`'s own default is `DEFAULT_EPSILON=0.05`, so the
# reported table would have silently carried ~5% epsilon-flipped routes.
# These tests make that regression impossible to reintroduce quietly: one
# static guard on the source (catches "someone changed the import back"),
# one behavioral guard that actually drives a call through and inspects the
# epsilon the underlying `retrieve()` received (catches "the import looks
# right but the wiring doesn't actually pin it").
# ---------------------------------------------------------------------------


class TestEvalPathAlwaysPinsEpsilonZero:
    def test_default_retriever_source_never_imports_bare_retrieve(self):
        """Static guard: the exact regression the audit found was `from
        medmemgraph.graph.retrieve import retrieve as real_retrieve`. Fail
        loudly if that import ever comes back, whether or not it happens to
        also be called correctly elsewhere."""
        source = inspect.getsource(_default_retriever)
        assert "retrieve_for_eval" in source
        assert "import retrieve as real_retrieve" not in source
        assert "import retrieve\n" not in source

    def test_default_retriever_returns_retrieve_for_eval_when_real_retrieve_enabled(self, monkeypatch):
        monkeypatch.setenv("MEDMEMGRAPH_USE_REAL_RETRIEVE", "1")
        from medmemgraph.graph.retrieve import retrieve_for_eval

        assert _default_retriever() is retrieve_for_eval

    def test_default_retriever_falls_back_to_mock_retrieve_when_flag_unset(self, monkeypatch):
        monkeypatch.delenv("MEDMEMGRAPH_USE_REAL_RETRIEVE", raising=False)
        from medmemgraph.contracts import mock_retrieve

        assert _default_retriever() is mock_retrieve

    def test_eval_path_never_reaches_the_real_retrieve_with_a_nonzero_epsilon(self, monkeypatch):
        """Behavioral guard, end to end: with the real-retrieve flag on,
        the retriever `_default_retriever()` hands back must drive
        `graph.retrieve.retrieve()` -- the actual function whose default is
        0.05 -- with epsilon exactly 0.0. Spies on the module-level
        `retrieve` name that `retrieve_for_eval` calls internally, so this
        fails if a future refactor of either function loses the pin."""
        monkeypatch.setenv("MEDMEMGRAPH_USE_REAL_RETRIEVE", "1")
        from medmemgraph.graph import retrieve as retrieve_mod

        seen_epsilons: list[float | None] = []

        def spy_retrieve(question, patient_id, k, *, epsilon=retrieve_mod.DEFAULT_EPSILON, **kwargs):
            seen_epsilons.append(epsilon)
            return RetrieveResult(items=[], route="vector", structural_absence=False, paths=[], latency_ms={})

        monkeypatch.setattr(retrieve_mod, "retrieve", spy_retrieve)

        retriever = _default_retriever()
        retriever("does the patient have any allergies?", "some-patient", 3)

        assert seen_epsilons == [0.0]


class TestTimestampExtractionMatchesRealProducers:
    """`_split_timestamp` against the text this project's item producers
    ACTUALLY emit.

    Regression for a bug found 2026-08-19: `_LEADING_TIMESTAMP_RE` requires the
    date to be the first characters inside the bracket, and NO producer emits
    that shape — turn units lead with an admission id, fact units with a fact
    id, graph units with a whole path expression. So `time` rendered as `null`
    on every evidence item and prose printed "time unknown", while the dates sat
    inside `text` unlabelled.

    That mattered because the three categories this system loses on
    (cross_admission_comparison, longitudinal_progression, frequency_pattern)
    are exactly the ones that need to order events in time — the reader was
    being told no timestamp existed."""

    def test_turn_unit_from_vector_index(self):
        # Exact format of graph/vector_index.py's turn unit.
        text = "[admission 20971116 turn 14 · 2120-08-06 20:15:00 · Doctor] We are adding spironolactone."
        time_, body = _split_timestamp(text)
        assert time_ == "2120-08-06 20:15:00"
        assert body == "We are adding spironolactone."

    def test_fact_unit_from_vector_index(self):
        text = "[fact abc123 · admission 20971116 · as of 2120-08-06T00:00:00] TAKES_MEDICATION furosemide"
        time_, body = _split_timestamp(text)
        assert time_ == "2120-08-06T00:00:00"
        assert body == "TAKES_MEDICATION furosemide"

    def test_graph_path_unit_keeps_its_text_intact(self):
        """A graph path's timestamp is embedded mid-expression, not a strippable
        prefix — surface the time but do not mangle the path."""
        text = "Symptom(headaches) <-[ABOUT]- Claim[REPORTS_SYMPTOM asserted, active, 2160-08-14T00:00:00..ongoing]"
        time_, body = _split_timestamp(text)
        assert time_ == "2160-08-14T00:00:00"
        assert body == text, "graph path text must survive unchanged"

    def test_legacy_ehr_rag_shape_still_works(self):
        """literature/15 R-QCC-045's template — kept working even though no
        producer here emits it."""
        time_, body = _split_timestamp("[2120-08-06 20:15:00] legacy shape")
        assert time_ == "2120-08-06 20:15:00"
        assert body == "legacy shape"

    def test_absent_timestamp_is_none_not_invented(self):
        time_, body = _split_timestamp("no timestamp anywhere in this text")
        assert time_ is None
        assert body == "no timestamp anywhere in this text"

    def test_rendered_json_carries_a_non_null_time(self):
        """The end-to-end symptom: what the model actually sees."""
        items = [
            RetrieveItem(
                text="[admission 20971116 turn 14 · 2120-08-06 20:15:00 · Doctor] Dose raised.",
                session_id="20971116", turn_ids=[14], score=0.9, channel="vector",
            )
        ]
        payload = json.loads(render_context(items, "json"))
        assert payload[0]["time"] == "2120-08-06 20:15:00", (
            "the reader must see a real timestamp, not null"
        )

    def test_rendered_prose_does_not_say_time_unknown(self):
        items = [
            RetrieveItem(
                text="[admission 20971116 turn 14 · 2120-08-06 20:15:00 · Doctor] Dose raised.",
                session_id="20971116", turn_ids=[14], score=0.9, channel="vector",
            )
        ]
        assert "time unknown" not in render_context(items, "prose")
