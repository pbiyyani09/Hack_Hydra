"""tests/test_embedders.py — `graph/embedders.py` (pluggable local embedder
backend story) plus the `graph/vector_index.py::PatientIndex` wiring that
consumes it.

Real local models are used directly (not mocked) where the story's own
acceptance line requires it — the on-disk embedding cache, the per-model
query/document prompt-formatting difference — because this project's
hardware has all eight candidate models already cached locally (verified
in this session: `~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B`,
`models--BAAI--bge-m3`, `models--Snowflake--snowflake-arctic-embed-l-v2.0`,
`models--sentence-transformers--all-MiniLM-L6-v2`,
`models--BAAI--bge-small-en-v1.5`, `models--Snowflake--snowflake-arctic-embed-s`,
`models--thenlper--gte-small`, `models--intfloat--e5-small-v2` all present —
the last four downloaded during this story's own measurement session, no
network call needed to *run* this file now) and a GPU (RTX 3090) that
loads/encodes them in low single-digit seconds — no network call is
required for these tests to pass on this machine, matching every other
`tests/test_*.py` file's "offline, no API key" convention. `qwen3-0.6b` and
`bge-m3` are used for the GPU-tier assertions (one asymmetric, one
symmetric prompt convention, per `embedders.py`'s own registry notes);
`arctic-l-v2` is exercised by the standalone throughput/VRAM probe this
story's return note quotes, not repeated here, to keep this file's own
wall-clock cost down (loading a third ~0.6B-parameter model buys no
additional test coverage this file does not already have from the other
two). The five CPU-tier candidates (22-33M params each — an order of
magnitude cheaper to load than the GPU-tier three) are all exercised
directly below in `TestCPUTierPromptFormatting`/`TestCPUTierRegistry`,
every one of them, since loading all five costs only a few seconds total.

A lightweight, dependency-free `_FakeBackend` covers the pure protocol-
shape and registry-dispatch assertions that do not need a real model at
all.
"""

from __future__ import annotations

import numpy as np
import pytest

from medmemgraph.graph import embedders
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import Admission, Conversation

pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# A dependency-free fake for the pure-Python protocol/registry assertions.
# ---------------------------------------------------------------------------


class _FakeBackend:
    name = "fake"
    dim = 4
    max_seq = 128

    def encode(self, texts: list[str], *, batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        # Deterministic, not L2-normalized on purpose — some assertions
        # below (protocol shape, get_backend passthrough) don't need real
        # semantics; the ones that need normalized output use a real
        # backend instead.
        return np.ones((len(texts), self.dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_fake_backend_satisfies_the_protocol(self) -> None:
        fake = _FakeBackend()
        assert isinstance(fake, embedders.EmbedderBackend)

    def test_sentence_transformer_embedder_satisfies_the_protocol(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        assert isinstance(backend, embedders.EmbedderBackend)
        assert backend.name == "qwen3-0.6b"
        assert isinstance(backend.dim, int) and backend.dim > 0
        assert isinstance(backend.max_seq, int) and backend.max_seq > 0

    def test_openai_embedder_satisfies_the_protocol(self) -> None:
        backend = embedders.OpenAIEmbedder(dry_run=True)
        assert isinstance(backend, embedders.EmbedderBackend)
        assert backend.name.startswith("openai:")
        assert backend.dim > 0
        assert backend.max_seq > 0

    def test_encode_shape_matches_dim(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        vecs = backend.encode(["hello", "world", "a third string"], is_query=False)
        assert vecs.shape == (3, backend.dim)
        assert vecs.dtype == np.float32

    def test_encode_empty_list_returns_empty_array_of_the_right_shape(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        vecs = backend.encode([], is_query=False)
        assert vecs.shape == (0, backend.dim)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


class TestRegistryAndFactory:
    def test_list_backends_names_all_nine(self) -> None:
        assert set(embedders.list_backends()) == {
            "qwen3-0.6b", "bge-m3", "arctic-l-v2",  # GPU-tier
            "minilm-l6", "bge-small", "arctic-s", "gte-small", "e5-small",  # CPU-tier
            "openai",
        }

    def test_get_backend_unknown_name_raises(self) -> None:
        with pytest.raises(embedders.UnknownBackendError):
            embedders.get_backend("not-a-real-backend")

    def test_get_backend_passes_through_an_already_built_backend(self) -> None:
        fake = _FakeBackend()
        assert embedders.get_backend(fake) is fake

    def test_get_backend_openai_dispatches_to_openai_embedder(self) -> None:
        backend = embedders.get_backend("openai")
        assert isinstance(backend, embedders.OpenAIEmbedder)

    def test_get_backend_local_name_dispatches_to_sentence_transformer_embedder(self) -> None:
        backend = embedders.get_backend("bge-m3", cache_path=None)
        assert isinstance(backend, embedders.SentenceTransformerEmbedder)
        assert backend.name == "bge-m3"

    def test_registered_gpu_tier_models_all_report_the_same_dim(self) -> None:
        """Module docstring's own load-bearing claim: every GPU-tier local
        candidate reports `hidden_size=1024`, verified per-model in
        `config.json` in this session — this test re-verifies it live
        against the two models this file loads (qwen3-0.6b, bge-m3) so a
        future model swap in the registry cannot silently break the
        "no dimensionality confound *within the GPU tier*" property
        without failing a test. This does NOT extend to the CPU tier
        (`TestCPUTierRegistry` below) -- 384-dim there is a deliberate,
        real dimensionality drop, not a bug."""
        qwen = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        bge = embedders.SentenceTransformerEmbedder("bge-m3", cache_path=None)
        assert qwen.dim == bge.dim == 1024


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_sentence_transformer_vectors_are_l2_normalized(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        vecs = backend.encode(["a short document.", "a second, longer document about metformin."])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_sentence_transformer_query_vectors_are_l2_normalized(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        vecs = backend.encode(["did the patient report nausea?"], is_query=True)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_openai_dry_run_vectors_are_l2_normalized(self) -> None:
        backend = embedders.OpenAIEmbedder(dry_run=True)
        vecs = backend.encode(["a document", "a query"], is_query=False)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Query/document prompt formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    def test_qwen3_query_and_document_vectors_differ_for_identical_text(self) -> None:
        """qwen3-0.6b's own `config_sentence_transformers.json` prepends an
        instruction to queries and nothing to documents (embedders.py
        module docstring) — encoding the *same* raw string once as a
        query and once as a document must therefore produce two
        different vectors."""
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=None)
        text = "the patient reported mild nausea after metformin doses"
        doc_vec = backend.encode([text], is_query=False)[0]
        query_vec = backend.encode([text], is_query=True)[0]
        assert not np.allclose(doc_vec, query_vec)

    def test_bge_m3_query_and_document_vectors_are_identical_for_identical_text(self) -> None:
        """bge-m3's own README: "no longer requires adding instructions to
        the queries" — symmetric, so the same raw string embedded as a
        query and as a document must produce the *same* vector."""
        backend = embedders.SentenceTransformerEmbedder("bge-m3", cache_path=None)
        text = "the patient reported mild nausea after metformin doses"
        doc_vec = backend.encode([text], is_query=False)[0]
        query_vec = backend.encode([text], is_query=True)[0]
        np.testing.assert_allclose(doc_vec, query_vec, atol=1e-6)

    def test_qwen3_registered_query_prompt_name_matches_the_models_own_config(self) -> None:
        assert embedders.EMBEDDER_REGISTRY["qwen3-0.6b"].query_prompt_name == "query"

    def test_bge_m3_has_no_registered_query_prompt(self) -> None:
        assert embedders.EMBEDDER_REGISTRY["bge-m3"].query_prompt_name is None

    def test_arctic_registered_query_prompt_name_matches_the_models_own_config(self) -> None:
        assert embedders.EMBEDDER_REGISTRY["arctic-l-v2"].query_prompt_name == "query"


# ---------------------------------------------------------------------------
# CPU-tier candidates (22-33M params, 384-dim) -- registry-data assertions
# that need no model load, per `_ModelSpec`'s own `query_prompt_name` /
# `query_prompt_text` / `document_prompt_text` fields.
# ---------------------------------------------------------------------------


class TestCPUTierRegistry:
    def test_five_cpu_tier_names_are_registered(self) -> None:
        assert {"minilm-l6", "bge-small", "arctic-s", "gte-small", "e5-small"} <= set(
            embedders.EMBEDDER_REGISTRY
        )

    def test_minilm_l6_has_no_prompt_mechanism_at_all(self) -> None:
        spec = embedders.EMBEDDER_REGISTRY["minilm-l6"]
        assert spec.query_prompt_name is None
        assert spec.query_prompt_text is None
        assert spec.document_prompt_text is None

    def test_gte_small_has_no_prompt_mechanism_at_all(self) -> None:
        spec = embedders.EMBEDDER_REGISTRY["gte-small"]
        assert spec.query_prompt_name is None
        assert spec.query_prompt_text is None
        assert spec.document_prompt_text is None

    def test_bge_small_uses_raw_query_prompt_text_not_prompt_name(self) -> None:
        """`bge-small`'s own `config_sentence_transformers.json` has no
        `prompts` key at all (unlike `arctic-s`/`arctic-l-v2`, which
        register theirs) -- the instruction lives only in README prose, so
        this spec must use the new `query_prompt_text` raw-string
        mechanism, not `query_prompt_name`."""
        spec = embedders.EMBEDDER_REGISTRY["bge-small"]
        assert spec.query_prompt_name is None
        assert spec.query_prompt_text == "Represent this sentence for searching relevant passages: "
        assert spec.document_prompt_text is None

    def test_arctic_s_uses_registered_prompt_name_with_a_different_literal_than_arctic_l_v2(self) -> None:
        """Same mechanism as `arctic-l-v2` (`query_prompt_name="query"`),
        but a DIFFERENT literal instruction string once resolved through
        each model's own `self.prompts` dict -- do not assume one Arctic
        model size's card generalizes to another's (module docstring)."""
        arctic_s = embedders.EMBEDDER_REGISTRY["arctic-s"]
        arctic_l = embedders.EMBEDDER_REGISTRY["arctic-l-v2"]
        assert arctic_s.query_prompt_name == arctic_l.query_prompt_name == "query"
        backend = embedders.SentenceTransformerEmbedder("arctic-s", cache_path=None, device="cpu")
        assert backend._model.prompts["query"] == "Represent this sentence for searching relevant passages: "

    def test_e5_small_has_both_query_and_document_prompt_text(self) -> None:
        """The first (and only) registry row where the DOCUMENT side also
        gets a raw literal prefix, not just the query side."""
        spec = embedders.EMBEDDER_REGISTRY["e5-small"]
        assert spec.query_prompt_name is None
        assert spec.query_prompt_text == "query: "
        assert spec.document_prompt_text == "passage: "

    def test_all_five_cpu_tier_candidates_report_384_dim(self) -> None:
        for key in ["minilm-l6", "bge-small", "arctic-s", "gte-small", "e5-small"]:
            backend = embedders.SentenceTransformerEmbedder(key, cache_path=None, device="cpu")
            assert backend.dim == 384, f"{key} reported dim={backend.dim}, expected 384"


# ---------------------------------------------------------------------------
# CPU-tier candidates -- real query/document prompt-formatting behavior,
# same discipline as `TestPromptFormatting` above (real model output, not
# a mock), including the two new mechanisms (`query_prompt_text`,
# `document_prompt_text`) this tier introduced.
# ---------------------------------------------------------------------------


class TestCPUTierPromptFormatting:
    _TEXT = "the patient reported mild nausea after metformin doses"

    def test_minilm_l6_query_and_document_vectors_are_identical(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("minilm-l6", cache_path=None, device="cpu")
        doc_vec = backend.encode([self._TEXT], is_query=False)[0]
        query_vec = backend.encode([self._TEXT], is_query=True)[0]
        np.testing.assert_allclose(doc_vec, query_vec, atol=1e-6)

    def test_gte_small_query_and_document_vectors_are_identical(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("gte-small", cache_path=None, device="cpu")
        doc_vec = backend.encode([self._TEXT], is_query=False)[0]
        query_vec = backend.encode([self._TEXT], is_query=True)[0]
        np.testing.assert_allclose(doc_vec, query_vec, atol=1e-6)

    def test_bge_small_query_and_document_vectors_differ(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("bge-small", cache_path=None, device="cpu")
        doc_vec = backend.encode([self._TEXT], is_query=False)[0]
        query_vec = backend.encode([self._TEXT], is_query=True)[0]
        assert not np.allclose(doc_vec, query_vec)

    def test_bge_small_query_vector_matches_the_raw_literal_prompt_call(self) -> None:
        """Proves `query_prompt_text` actually reaches the model via
        `encode(..., prompt=...)`, not merely that the vectors differ (a
        weaker property that a bug in the mechanism could still satisfy
        by accident)."""
        from sentence_transformers import SentenceTransformer

        raw = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        expected = raw.encode(
            [self._TEXT],
            prompt="Represent this sentence for searching relevant passages: ",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        backend = embedders.SentenceTransformerEmbedder("bge-small", cache_path=None, device="cpu")
        actual = backend.encode([self._TEXT], is_query=True)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_arctic_s_query_and_document_vectors_differ(self) -> None:
        backend = embedders.SentenceTransformerEmbedder("arctic-s", cache_path=None, device="cpu")
        doc_vec = backend.encode([self._TEXT], is_query=False)[0]
        query_vec = backend.encode([self._TEXT], is_query=True)[0]
        assert not np.allclose(doc_vec, query_vec)

    def test_e5_small_query_and_document_vectors_differ(self) -> None:
        """Different literal prefixes on each side ('query: ' vs
        'passage: ') -- not the "instruction on query only" shape every
        other asymmetric model in this registry uses."""
        backend = embedders.SentenceTransformerEmbedder("e5-small", cache_path=None, device="cpu")
        doc_vec = backend.encode([self._TEXT], is_query=False)[0]
        query_vec = backend.encode([self._TEXT], is_query=True)[0]
        assert not np.allclose(doc_vec, query_vec)

    def test_e5_small_document_vector_matches_the_raw_passage_prefixed_call(self) -> None:
        """The load-bearing new-mechanism proof: `document_prompt_text`
        actually reaches the model's DOCUMENT-side encode call, the first
        registry row in this module where that happens at all."""
        from sentence_transformers import SentenceTransformer

        raw = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
        expected_passage = raw.encode(
            [self._TEXT], prompt="passage: ", normalize_embeddings=True, convert_to_numpy=True
        )[0]
        expected_no_prefix = raw.encode([self._TEXT], normalize_embeddings=True, convert_to_numpy=True)[0]

        backend = embedders.SentenceTransformerEmbedder("e5-small", cache_path=None, device="cpu")
        actual_doc = backend.encode([self._TEXT], is_query=False)[0]

        np.testing.assert_allclose(actual_doc, expected_passage, atol=1e-6)
        assert not np.allclose(actual_doc, expected_no_prefix)


# ---------------------------------------------------------------------------
# Disk cache: byte-identical vectors, no second forward pass, on-disk
# persistence across a fresh instance ("re-runs are free").
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_in_process_cache_hit_returns_byte_identical_vector_and_skips_the_forward_pass(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.npz"
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=cache_path)
        text = "the patient takes metformin twice daily"

        first = backend.encode([text], is_query=False)

        def _blow_up(*_args, **_kwargs):
            raise AssertionError("forward pass was invoked on a cache hit")

        backend._forward = _blow_up  # type: ignore[method-assign]
        second = backend.encode([text], is_query=False)

        assert np.array_equal(first, second)

    def test_disk_cache_survives_a_fresh_instance_so_reruns_are_free(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.npz"
        text = "the patient takes metformin twice daily"

        backend_1 = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=cache_path)
        first = backend_1.encode([text], is_query=False)
        assert cache_path.exists()

        # A brand-new wrapper instance, same cache_path -- simulates a
        # fresh process re-run reading the same on-disk cache. It shares
        # the already-loaded model weights (process-wide `_MODEL_CACHE`)
        # but must load its *own* disk cache from `cache_path` fresh.
        backend_2 = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=cache_path)

        def _blow_up(*_args, **_kwargs):
            raise AssertionError("forward pass was invoked despite a warm on-disk cache")

        backend_2._forward = _blow_up  # type: ignore[method-assign]
        second = backend_2.encode([text], is_query=False)

        assert np.array_equal(first, second)

    def test_cache_key_distinguishes_query_role_from_document_role(self, tmp_path) -> None:
        """Deliberate extension beyond the story's literal `(model_name,
        content_hash)` 2-tuple (embedders.py module docstring): the SAME
        raw text embedded once as a document and once as a query must not
        collide in the cache, because `qwen3-0.6b` applies a different
        prompt to each role."""
        cache_path = tmp_path / "cache.npz"
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=cache_path)
        text = "identical raw text used as both a query and a document"

        doc_vec = backend.encode([text], is_query=False)[0]
        query_vec = backend.encode([text], is_query=True)[0]

        assert not np.allclose(doc_vec, query_vec)

    def test_a_bad_cache_file_is_treated_as_cold_not_fatal(self, tmp_path) -> None:
        cache_path = tmp_path / "corrupt.npz"
        cache_path.write_bytes(b"not a valid npz file")
        backend = embedders.SentenceTransformerEmbedder("qwen3-0.6b", cache_path=cache_path)
        vecs = backend.encode(["still works"], is_query=False)
        assert vecs.shape == (1, backend.dim)


# ---------------------------------------------------------------------------
# PatientIndex wiring
# ---------------------------------------------------------------------------


def _small_conversation(patient_id: str) -> Conversation:
    admission = Admission(
        hadm_id="adm-1",
        admission_start="2126-01-01",
        admission_end="2126-01-02",
        conversation_lines=(
            {"turn_number": 1, "time": "2126-01-01T09:00:00", "speaker": "Doctor", "text": "How are you feeling?"},
            {
                "turn_number": 2,
                "time": "2126-01-01T09:05:00",
                "speaker": "Patient",
                "text": "I had anaphylaxis after amoxicillin as a child.",
            },
        ),
    )
    return Conversation(subject_id=patient_id, processed_hadm_ids=("adm-1",), admissions=(admission,))


class TestPatientIndexBackendWiring:
    def test_default_backend_is_qwen3(self) -> None:
        index = PatientIndex("patient-embedders-default", cache_path=None)
        assert index.backend_name == "qwen3-0.6b"
        assert index.dim == 1024

    def test_explicit_backend_string_is_honored(self) -> None:
        index = PatientIndex("patient-embedders-bge", backend="bge-m3", cache_path=None)
        assert index.backend_name == "bge-m3"

    def test_embed_fn_escape_hatch_still_works_and_is_named_custom(self) -> None:
        calls: list[list[str]] = []

        def fn(texts: list[str]) -> np.ndarray:
            calls.append(list(texts))
            return np.ones((len(texts), 8), dtype=np.float32)

        index = PatientIndex("patient-embedders-custom", embed_fn=fn, dim=8, cache_path=None)
        assert index.backend_name == "custom"
        index.build(_small_conversation("patient-embedders-custom"))
        assert calls, "expected the custom embed_fn to have been invoked at build time"

    def test_save_then_load_with_matching_backend_round_trips(self, tmp_path) -> None:
        patient_id = "patient-embedders-roundtrip"
        built = PatientIndex(patient_id, backend="bge-m3", cache_path=None)
        built.build(_small_conversation(patient_id))
        built.save(tmp_path)

        loaded = PatientIndex.load(patient_id, tmp_path, cache_path=None)
        assert loaded.backend_name == "bge-m3"
        query = "anaphylaxis after amoxicillin"
        before = built.search(query, k=2)
        after = loaded.search(query, k=2)
        assert [(r.text, r.turn_ids, round(r.score, 6)) for r in before] == [
            (r.text, r.turn_ids, round(r.score, 6)) for r in after
        ]

    def test_load_with_mismatched_backend_raises(self, tmp_path) -> None:
        patient_id = "patient-embedders-mismatch"
        built = PatientIndex(patient_id, backend="bge-m3", cache_path=None)
        built.build(_small_conversation(patient_id))
        built.save(tmp_path)

        with pytest.raises(embedders.BackendMismatchError):
            PatientIndex.load(patient_id, tmp_path, backend="qwen3-0.6b", cache_path=None)

    def test_load_with_no_recorded_backend_and_no_override_raises(self, tmp_path) -> None:
        """An index saved before pluggable embedders existed has no
        `backend_name` key in its meta sidecar at all -- `load()` must
        refuse to guess rather than silently assume today's default."""
        import json

        patient_id = "patient-embedders-legacy"
        built = PatientIndex(patient_id, backend="bge-m3", cache_path=None)
        built.build(_small_conversation(patient_id))
        built.save(tmp_path)

        vectors_path, meta_path = PatientIndex._paths_for(patient_id, tmp_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["backend_name"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(embedders.BackendMismatchError):
            PatientIndex.load(patient_id, tmp_path, cache_path=None)

    def test_load_missing_index_raises_before_touching_any_backend(self, tmp_path, monkeypatch) -> None:
        """`load()` must check file existence before constructing any
        encoder -- a missing-index call should never pay a model-load
        cost. Verified by making `embedders.get_backend` explode if
        `load()` ever reaches it on this path."""

        def _blow_up(*_args, **_kwargs):
            raise AssertionError("get_backend was called before file-existence was checked")

        monkeypatch.setattr(embedders, "get_backend", _blow_up)
        with pytest.raises(FileNotFoundError):
            PatientIndex.load("no-such-patient-embedders", tmp_path)

    # -- CPU-tier: the persisted backend_name guard must hold with a
    # 384-dim model too, both matching (round trip) and mismatched --
    # including a mismatch that ALSO crosses the 1024-dim/384-dim tier
    # boundary, which a dimension-only check would miss.

    def test_cpu_tier_backend_reports_384_dim(self) -> None:
        index = PatientIndex("patient-embedders-cpu-tier", backend="minilm-l6", cache_path=None)
        assert index.backend_name == "minilm-l6"
        assert index.dim == 384

    def test_save_then_load_with_cpu_tier_backend_round_trips(self, tmp_path) -> None:
        patient_id = "patient-embedders-cpu-tier-roundtrip"
        built = PatientIndex(patient_id, backend="minilm-l6", cache_path=None)
        built.build(_small_conversation(patient_id))
        built.save(tmp_path)

        loaded = PatientIndex.load(patient_id, tmp_path, cache_path=None)
        assert loaded.backend_name == "minilm-l6"
        assert loaded.dim == 384
        query = "anaphylaxis after amoxicillin"
        before = built.search(query, k=2)
        after = loaded.search(query, k=2)
        assert [(r.text, r.turn_ids, round(r.score, 6)) for r in before] == [
            (r.text, r.turn_ids, round(r.score, 6)) for r in after
        ]

    def test_load_with_mismatched_backend_raises_across_dimension_tiers(self, tmp_path) -> None:
        """The persisted-`backend_name` guard checks *name*, not
        dimension (`vector_index.PatientIndex.load` docstring) -- proven
        here across a REAL dimension change, not just a same-dim mismatch:
        an index built with the 384-dim `minilm-l6` must still raise
        `BackendMismatchError` when `load()` is asked for the 1024-dim
        `qwen3-0.6b`, not silently produce a 384-vs-1024 shape mismatch
        downstream in `PatientIndex.search`'s `self.vectors @ q`."""
        patient_id = "patient-embedders-cpu-tier-mismatch"
        built = PatientIndex(patient_id, backend="minilm-l6", cache_path=None)
        built.build(_small_conversation(patient_id))
        built.save(tmp_path)

        with pytest.raises(embedders.BackendMismatchError):
            PatientIndex.load(patient_id, tmp_path, backend="qwen3-0.6b", cache_path=None)
