"""Offline split invariants. No corpus text is opened."""

from __future__ import annotations

import random

import pytest

from medmemgraph.eval.rerank_split import EVAL_TRIO, build_split


def test_eval_trio_forced_and_no_overlap(monkeypatch):
    fake = [
        "aaaaaa01",
        "10056223",  # trio in the middle of sort
        "bbbbbb02",
        "10213338",
        "cccccc03",
        "10312715",
        "dddddd04",
        "eeeeee05",
        "ffffff06",
        "gggggg07",
        "hhhhhh08",
        "iiiiii09",
        "jjjjjj10",
        "kkkkkk11",
        "llllll12",
    ]
    monkeypatch.setattr(
        "medmemgraph.eval.rerank_split.list_patients", lambda root=None: sorted(fake)
    )
    split = build_split(n_dev=10, seed=20260817)
    assert set(split["eval"]) == set(EVAL_TRIO)
    assert split["n_dev"] == 10
    assert len(split["dev"]) == 10
    assert set(split["eval"]).isdisjoint(split["dev"])
    assert set(split["eval"]).isdisjoint(split["train"])
    assert set(split["dev"]).isdisjoint(split["train"])
    assigned = sorted(split["eval"] + split["dev"] + split["train"])
    assert assigned == sorted(fake)


def test_split_is_deterministic(monkeypatch):
    remaining = [f"p{i:03d}" for i in range(20)]
    patients = list(EVAL_TRIO) + remaining
    monkeypatch.setattr(
        "medmemgraph.eval.rerank_split.list_patients", lambda root=None: sorted(patients)
    )
    a = build_split()
    b = build_split()
    assert a["train"] == b["train"]
    assert a["dev"] == b["dev"]
    expected_dev = sorted(random.Random(20260817).sample(sorted(remaining), 10))
    assert a["dev"] == expected_dev


def test_build_split_raises_if_eval_id_missing(monkeypatch):
    fake = ["aaaaaa01", "10056223", "10213338"]  # 10312715 omitted
    monkeypatch.setattr(
        "medmemgraph.eval.rerank_split.list_patients", lambda root=None: sorted(fake)
    )
    with pytest.raises(RuntimeError, match="eval trio"):
        build_split()
