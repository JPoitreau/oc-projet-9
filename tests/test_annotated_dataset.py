"""Validation du schéma du jeu de test annoté (sans métriques Ragas)."""

import json
from pathlib import Path

import pytest

ANNOTATED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "evaluation" / "annotated_qa.json"
)

ALLOWED_CATEGORIES = {
    "factual",
    "ambiguous",
    "out_of_scope",
    "retrieval_trap",
}


@pytest.mark.skipif(not ANNOTATED_PATH.exists(), reason="Jeu annoté non fourni")
def test_annotated_dataset_schema():
    data = json.loads(ANNOTATED_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 10
    for item in data:
        assert isinstance(item, dict)
        assert item.get("question", "").strip()
        assert item.get("reference_answer", "").strip()
        category = item.get("category")
        if category is not None:
            assert category in ALLOWED_CATEGORIES
