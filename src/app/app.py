import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from utils.utils import get_embeddings, search_relevant_segments, load_index_and_raw  # noqa: F401

load_dotenv()

streamlit_role_mapping = {
    "ai": "assistant",
    "human": "user",
    "system": "system",
}

model = init_chat_model(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

#if "messages" not in st.session_state:
#    st.session_state.messages = [AIMessage(content="Bonjour, je suis l'assistant virtuel de renseignement sur les événements publics. Comment puis-je vous aider?")]

if "messages" not in st.session_state:
    st.session_state.messages = []

def build_prompt_session(messages, raw, index, question=None, max_messages=5):
    """
    Construit un prompt enrichi avec les segments pertinents et l'historique récent.
    """
    # Limiter le nombre de messages récents
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Création du système prompt
    system_prompt = """
        Vous êtes l'assistant officiel de l'entreprise Puls-Event, une entreprise technologique 
        qui développe une plateforme de recommandations culturelles personnalisées.
        Votre mission est de répondre aux questions utilisateurs sur les évènements publics passés et à venir dans la ville de Toulouse.
        Source d'information autorisée:
        - Vous devez vous baser EXCLUSIVEMENT sur les données fournies par la base de connaissances.
        Comportements obligatoires:
        - Adoptez un ton formel, courtois et accessible à tous les publics.
        - Face à une question ambiguë, demandez systématiquement des précisions.
        - Si une information n'est pas disponible dans votre base de connaissances, indiquez-le clairement.
        - Si possible, proposez une alternative cohérente en cas d'information manquante.
        - Adaptez votre niveau de détail à la complexité de la demande.
        Comportements interdits:
        - Ne fournissez JAMAIS d'informations dont vous n'êtes pas certain.
        - N'inventez pas d'évènements, lieux ou dates qui ne sont pas dans la base de connaissances.
        """

        # Si une question est fournie, rechercher les segments pertinents
    context_segments = []
    if question:
        context_segments = search_relevant_segments(raw,index,question)
        formatted = [
        f"--- Événement {i + 1} ---\n{segment}"
        for i, segment in enumerate(context_segments)
        ]


        # Ajout du contexte des segments pertinents
    if context_segments:
        system_prompt += "Veuillez utiliser les informations suivantes pour répondre à la question:\n\n"
        system_prompt += "\n\n".join(formatted)
        system_prompt += "\n\nSi les informations fournies ne sont pas suffisantes pour répondre précisément, veuillez l'indiquer."

    # Suppression des messages système existants
    recent_messages = [m for m in recent_messages if m.type != "system"]

    # Ajout du système prompt mis à jour avec le contexte des segments pertinents
    recent_messages.insert(0, SystemMessage(content=system_prompt))

    return recent_messages

def generate_response(prompt_messages):
    try:
        response = model.invoke(prompt_messages)
        return response.content
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération de la réponse: {e}")
        return "Je suis désolé, j'ai rencontré un problème technique. Veuillez réessayer."


@st.cache_resource
def get_rag_resources():
    index, raw = load_index_and_raw()
    return index, raw

index, raw = get_rag_resources()

st.title("Assistant de renseignement sur les événements publics")
for message in st.session_state.messages:
    with st.chat_message(streamlit_role_mapping[message.type]):
        st.write(message.content)

if prompt := st.chat_input("Comment puis-je vous aider?"):
    # Ajout du message de l'utilisateur à l'historique
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Affichage du message de l'utilisateur
    with st.chat_message("user"):
        st.write(prompt)

    # Préparation du prompt avec l'historique
    prompt_messages = build_prompt_session(st.session_state.messages, raw, index, prompt)

    # Affichage d'un indicateur de chargement
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("En train de réfléchir...")
        # Génération de la réponse
        response = generate_response(prompt_messages)
        # Affichage de la réponse
        message_placeholder.write(response)
        
    # Ajout de la réponse à l'historique
    st.session_state.messages.append(AIMessage(content=response))

#if st.button("Delete conversation"):
#    st.session_state.messages = [AIMessage(content="Bonjour, je suis l'assistant virtuel de renseignement sur les événements publics. Comment puis-je vous aider?")]
#    st.rerun()

if st.button("Delete conversation"):
    #st.session_state.messages = [AIMessage(content="Notre actualité présipotère maintenant:")]
    st.session_state.messages = []
    st.rerun()