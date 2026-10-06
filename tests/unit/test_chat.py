"""Tests unitaires pour la construction du prompt et la génération de réponse."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.chat import build_prompt_session, generate_response, run_rag_query


def test_build_prompt_session_includes_context_and_system(mini_faiss_index):
    index, raw = mini_faiss_index
    fake_segments = ["title_fr: Atelier piano", "title_fr: Atelier numerique"]
    messages = [HumanMessage(content="Bonjour")]

    with patch("app.chat.search_relevant_segments", return_value=fake_segments):
        prompt = build_prompt_session(messages, raw, index, question="piano")

    assert isinstance(prompt[0], SystemMessage)
    system_content = prompt[0].content
    assert "Puls-Event" in system_content
    assert "Toulouse" in system_content
    assert "Atelier piano" in system_content


def test_build_prompt_session_uses_provided_context_without_search(mini_faiss_index):
    index, raw = mini_faiss_index
    preset = ["title_fr: Événement preset"]

    with patch("app.chat.search_relevant_segments") as mock_search:
        prompt = build_prompt_session(
            [],
            raw,
            index,
            question="test",
            context_segments=preset,
        )

    mock_search.assert_not_called()
    assert "Événement preset" in prompt[0].content


def test_run_rag_query(mini_faiss_index):
    index, raw = mini_faiss_index
    captured = []

    def capture_generate(msgs):
        captured.append(msgs)
        return "Réponse RAG"

    with patch(
        "app.chat.search_relevant_segments",
        return_value=["ctx"],
    ):
        with patch("app.chat.generate_response", side_effect=capture_generate):
            answer, contexts = run_rag_query("Question test", raw, index)

    assert answer == "Réponse RAG"
    assert contexts == ["ctx"]
    assert captured[0][-1].content == "Question test"


def test_build_prompt_session_truncates_history(mini_faiss_index):
    index, raw = mini_faiss_index
    messages = [HumanMessage(content=f"msg-{i}") for i in range(10)]

    with patch("app.chat.search_relevant_segments", return_value=[]):
        prompt = build_prompt_session(
            messages, raw, index, question="test", max_messages=3
        )

    non_system = [m for m in prompt if m.type != "system"]
    assert len(non_system) == 3
    assert non_system[-1].content == "msg-9"


def test_generate_response_success():
    mock_response = MagicMock()
    mock_response.content = "Réponse générée"
    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_response
    with patch("app.chat.model", mock_model):
        result = generate_response([HumanMessage(content="Question")])
    assert result == "Réponse générée"


def test_generate_response_on_error():
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("API down")
    with patch("app.chat.model", mock_model):
        result = generate_response([HumanMessage(content="Question")])
    assert "problème technique" in result.lower()
