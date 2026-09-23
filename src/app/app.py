import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.chat import build_prompt_session, generate_response
from utils.utils import load_index_and_raw, DATA_DIR

if __name__ == "__main__":
    streamlit_role_mapping = {
        "ai": "assistant",
        "human": "user",
        "system": "system",
    }

    if "messages" not in st.session_state:
        st.session_state.messages = []

    index_path = DATA_DIR / "faiss_index.idx"
    metadata_path = DATA_DIR / "metadata.csv"

    @st.cache_resource
    def get_rag_resources():
        return load_index_and_raw()

    if not index_path.exists() or not metadata_path.exists():
        st.warning(
            "L'index de recherche n'est pas disponible. "
            "Lancez la création de l'index (script db_creation ou endpoint /rebuild de l'API)."
        )
        index, raw = None, None
    else:
        index, raw = get_rag_resources()

    st.title("Assistant de renseignement sur les événements publics")
    for message in st.session_state.messages:
        with st.chat_message(streamlit_role_mapping[message.type]):
            st.write(message.content)

    if prompt := st.chat_input("Comment puis-je vous aider?"):
        if index is None or raw is None:
            st.error("Index non chargé. Veuillez reconstruire l'index avant de poser une question.")
        else:
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)

            prompt_messages = build_prompt_session(
                st.session_state.messages, raw, index, prompt
            )

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("En train de réfléchir...")
                response = generate_response(prompt_messages)
                message_placeholder.write(response)

            st.session_state.messages.append(AIMessage(content=response))

    if st.button("Delete conversation"):
        st.session_state.messages = []
        st.rerun()
