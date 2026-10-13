"""Tests fonctionnels de l'API FastAPI (RAG)."""

import importlib
from unittest.mock import patch

import app.api as api_module


def test_openapi_docs(api_client):
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()


def test_ask_success(api_client, mini_faiss_index, patch_get_embeddings):
    index, raw = mini_faiss_index
    api_module.index = index
    api_module.raw = raw

    with patch("app.api.generate_response", return_value="Réponse mock"):
        response = api_client.post(
            "/ask/",
            json={"user_question": "Y a-t-il des cours de piano ?"},
        )

    assert response.status_code == 200
    assert response.json() == "Réponse mock"


def test_ask_without_index(api_client):
    api_module.index = None
    api_module.raw = None

    response = api_client.post(
        "/ask/",
        json={"user_question": "Question test"},
    )
    assert response.status_code == 503


def test_ask_empty_question(api_client, mini_faiss_index):
    index, raw = mini_faiss_index
    api_module.index = index
    api_module.raw = raw

    response = api_client.post("/ask/", json={"user_question": "   "})
    assert response.status_code == 422


def test_ask_invalid_body(api_client):
    response = api_client.post("/ask/", json={})
    assert response.status_code == 422


def test_rebuild_success(api_client, mini_faiss_index):
    index, raw = mini_faiss_index
    rebuild_payload = {
        "nb of indexed documents": 2,
        "dimension of the embeddings": 8,
        "index path": "/tmp/faiss_index.idx",
        "metadata path": "/tmp/metadata.csv",
    }

    with patch("app.api.rebuild_faiss_index", return_value=rebuild_payload), patch("app.api.load_index_and_raw", return_value=(index, raw)):
        response = api_client.get("/rebuild/")

    assert response.status_code == 200
    assert response.json()["nb of indexed documents"] == 2


def test_rebuild_missing_data(api_client):
    with patch(
        "app.api.rebuild_faiss_index",
        side_effect=FileNotFoundError("CSV absent"),
    ):
        response = api_client.get("/rebuild/")

    assert response.status_code == 503
    assert "CSV absent" in response.json()["detail"]


def test_ask_with_message_roles(api_client, mini_faiss_index, patch_get_embeddings):
    index, raw = mini_faiss_index
    api_module.index = index
    api_module.raw = raw

    payload = {
        "user_question": "Question avec historique",
        "messages": [
            {"role": "human", "content": "Bonjour"},
            {"role": "ai", "content": "Bonjour, comment puis-je vous aider ?"},
            {"role": "system", "content": "Contexte système"},
        ],
    }
    with patch("app.api.generate_response", return_value="Réponse roles"):
        response = api_client.post("/ask/", json=payload)

    assert response.status_code == 200
    assert response.json() == "Réponse roles"


def test_ask_swagger_placeholder_message(api_client, mini_faiss_index, patch_get_embeddings):
    index, raw = mini_faiss_index
    api_module.index = index
    api_module.raw = raw

    with patch("app.api.generate_response", return_value="OK"):
        response = api_client.post(
            "/ask/",
            json={
                "user_question": "Test placeholder",
                "messages": [{"role": "human", "content": "string"}],
            },
        )

    assert response.status_code == 200


def test_api_module_without_index_files(tmp_path, monkeypatch):
    import utils.utils as utils_module

    empty_dir = tmp_path / "processed"
    empty_dir.mkdir(parents=True)
    monkeypatch.setattr(utils_module, "DATA_DIR", empty_dir)
    reloaded = importlib.reload(api_module)
    assert reloaded.index is None
    assert reloaded.raw is None
