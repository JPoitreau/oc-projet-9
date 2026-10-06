"""Fixtures partagées pour la suite de tests (sans appels OpenAI)."""

import hashlib
import struct
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV_PATH = FIXTURES_DIR / "sample_events.csv"
EMBEDDING_DIM = 8


def fake_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Vecteur déterministe à partir du texte (reproductible en CI)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        chunk = digest[i * 4 : (i + 1) * 4]
        if len(chunk) < 4:
            chunk = (chunk + digest)[:4]
        values.append(struct.unpack("!f", chunk)[0])
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


@pytest.fixture
def sample_csv_path():
    return SAMPLE_CSV_PATH


@pytest.fixture
def sample_raw_df(sample_csv_path):
    return pd.read_csv(sample_csv_path, sep=";")


@pytest.fixture
def fake_get_embeddings():
    return lambda text, model="text-embedding-3-small": fake_embedding(text)


@pytest.fixture
def mini_faiss_index(sample_raw_df, fake_get_embeddings):
    """Index FAISS en mémoire aligné sur les uid du mini CSV (descriptions valides uniquement)."""
    from app.db_creation import clean_text

    clean_description = (
        sample_raw_df["description_fr"]
        .dropna()
        .apply(clean_text)
        .dropna()
    )
    raw = sample_raw_df[sample_raw_df.index.isin(clean_description.index)]
    documents = clean_description.tolist()
    ids = np.array(raw.uid.tolist(), dtype=np.int64)
    embeddings = np.array(
        [fake_get_embeddings(doc) for doc in documents], dtype=np.float32
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    index.add_with_ids(embeddings, ids)
    return index, raw


@pytest.fixture
def patch_get_embeddings(monkeypatch, fake_get_embeddings):
    """Remplace get_embeddings dans utils et db_creation."""
    import app.db_creation as db_creation
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(db_creation, "get_embeddings", fake_get_embeddings)


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from app.api import app

    return TestClient(app)
