"""Tests unitaires pour les utilitaires RAG (FAISS, formatage)."""

import time

import faiss
import numpy as np
import pandas as pd
import pytest

from utils.utils import (
    format_row_as_key_value,
    load_index_and_raw,
    search_relevant_segments,
)


def test_format_row_as_key_value_skips_empty():
    row = pd.Series({"title_fr": "Concert", "description_fr": "", "uid": 1})
    text = format_row_as_key_value(row)
    assert "title_fr: Concert" in text
    assert "description_fr" not in text


def test_search_relevant_segments(mini_faiss_index, patch_get_embeddings):
    index, raw = mini_faiss_index
    segments = search_relevant_segments(raw, index, "atelier piano", k=3)
    assert 1 <= len(segments) <= 3
    assert any("description_fr" in s for s in segments)


def test_load_index_and_raw(tmp_path, mini_faiss_index, monkeypatch):
    import utils.utils as utils_module

    index, raw = mini_faiss_index
    processed = tmp_path / "processed"
    processed.mkdir()
    index_path = processed / "faiss_index.idx"
    metadata_path = processed / "metadata.csv"
    faiss.write_index(index, str(index_path))
    raw.to_csv(metadata_path, index=False)

    monkeypatch.setattr(utils_module, "DATA_DIR", processed)

    loaded_index, loaded_raw = load_index_and_raw()
    assert loaded_index.ntotal == index.ntotal
    assert len(loaded_raw) == len(raw)


def test_load_index_and_raw_missing_raises(tmp_path, monkeypatch):
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "DATA_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        load_index_and_raw()


def test_load_index_and_raw_missing_metadata_raises(
    tmp_path, mini_faiss_index, monkeypatch
):
    import utils.utils as utils_module

    index, _ = mini_faiss_index
    processed = tmp_path / "processed"
    processed.mkdir()
    faiss.write_index(index, str(processed / "faiss_index.idx"))
    monkeypatch.setattr(utils_module, "DATA_DIR", processed)

    with pytest.raises(FileNotFoundError):
        load_index_and_raw()


def test_search_relevant_segments_skips_unknown_uid(mini_faiss_index, patch_get_embeddings):
    _index, raw = mini_faiss_index
    bogus = faiss.IndexIDMap(faiss.IndexFlatL2(8))
    bogus.add_with_ids(
        np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        np.array([99999], dtype=np.int64),
    )
    segments = search_relevant_segments(raw, bogus, "atelier piano", k=1)
    assert segments == []


@pytest.mark.slow
def test_faiss_search_performance(mini_faiss_index):
    index, _ = mini_faiss_index
    query = np.random.rand(1, 8).astype("float32")
    start = time.perf_counter()
    for _ in range(100):
        index.search(query, 3)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
