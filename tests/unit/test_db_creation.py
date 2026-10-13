"""Tests unitaires pour le nettoyage, le chunking et la reconstruction d'index."""

import shutil

import faiss
import pandas as pd
import pytest

from app.db_creation import (
    check_for_dry_emails_urls_or_numeric,
    clean_text,
    normalize_text,
    rebuild_faiss_index,
    split_text_into_chunks,
)

REQUIRED_COLUMNS = {"uid", "description_fr", "title_fr", "location_city"}


def test_fixture_csv_schema(sample_csv_path):
    raw = pd.read_csv(sample_csv_path, sep=";")
    assert REQUIRED_COLUMNS.issubset(raw.columns)
    assert raw["uid"].is_unique
    assert raw["description_fr"].notna().any()
    cities = raw["location_city"].dropna().unique()
    assert all(str(c) == "Toulouse" for c in cities)


def test_check_for_dry_emails_urls_or_numeric():
    assert check_for_dry_emails_urls_or_numeric("https://example.com")["valid"] is False
    assert check_for_dry_emails_urls_or_numeric("user@domain.com")["valid"] is False
    assert check_for_dry_emails_urls_or_numeric("12345")["valid"] is False
    assert check_for_dry_emails_urls_or_numeric("Atelier piano à Toulouse")["valid"] is True


def test_normalize_text():
    assert normalize_text("  HELLO   WORLD  ") == "hello world"
    assert normalize_text("foo-bar") == "foo bar"
    assert "!" not in normalize_text("Salut !")
    assert normalize_text("de piano") == "piano"


def test_clean_text_invalid_returns_none():
    assert clean_text("https://example.com") is None
    assert clean_text("Atelier culturel") is not None


def test_split_text_into_chunks():
    text = "mot " * 500
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_rebuild_faiss_index(tmp_path, patch_get_embeddings, monkeypatch, sample_csv_path):
    import utils.utils as utils_module
    from app import db_creation

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dest = raw_dir / "evenements-publics-openagenda.csv"
    shutil.copy(sample_csv_path, dest)

    monkeypatch.setattr(utils_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(utils_module, "RAW_DATA_PATH", dest)
    monkeypatch.setattr(db_creation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(db_creation, "RAW_DATA_PATH", dest)

    result = rebuild_faiss_index()

    assert result["nb of indexed documents"] == 2
    assert result["dimension of the embeddings"] == 8
    index_path = tmp_path / "data" / "processed" / "faiss_index.idx"
    metadata_path = tmp_path / "data" / "processed" / "metadata.csv"
    assert index_path.exists()
    assert metadata_path.exists()
    index = faiss.read_index(str(index_path))
    assert index.ntotal == 2


@pytest.mark.slow
def test_rebuild_then_ask_pipeline(
    tmp_path, patch_get_embeddings, monkeypatch, sample_csv_path, fake_get_embeddings
):
    """Pipeline mini rebuild + recherche sans LLM."""
    import utils.utils as utils_module
    from app import db_creation
    from utils.utils import search_relevant_segments

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dest = raw_dir / "evenements-publics-openagenda.csv"
    shutil.copy(sample_csv_path, dest)

    monkeypatch.setattr(utils_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(utils_module, "RAW_DATA_PATH", dest)
    monkeypatch.setattr(db_creation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(db_creation, "RAW_DATA_PATH", dest)

    rebuild_faiss_index()
    monkeypatch.setattr(utils_module, "DATA_DIR", tmp_path / "data" / "processed")
    index, raw = utils_module.load_index_and_raw()
    assert index.d == len(fake_get_embeddings("probe"))
    segments = search_relevant_segments(raw, index, "cours de piano", k=2)
    assert len(segments) >= 1
    assert any("piano" in s.lower() for s in segments)
