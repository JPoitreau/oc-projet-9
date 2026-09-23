import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from utils.utils import search_relevant_segments

load_dotenv()

logger = logging.getLogger(__name__)

model = init_chat_model(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def build_prompt_session(messages, raw, index, question=None, max_messages=5):
    """
    Construit un prompt enrichi avec les segments pertinents et l'historique récent.
    """
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

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

    context_segments = []
    if question:
        context_segments = search_relevant_segments(raw, index, question)
        formatted = [
            f"--- Événement {i + 1} ---\n{segment}"
            for i, segment in enumerate(context_segments)
        ]

    if context_segments:
        system_prompt += "Veuillez utiliser les informations suivantes pour répondre à la question:\n\n"
        system_prompt += "\n\n".join(formatted)
        system_prompt += "\n\nSi les informations fournies ne sont pas suffisantes pour répondre précisément, veuillez l'indiquer."

    recent_messages = [m for m in recent_messages if m.type != "system"]
    recent_messages.insert(0, SystemMessage(content=system_prompt))

    return recent_messages


def generate_response(prompt_messages):
    try:
        response = model.invoke(prompt_messages)
        return response.content
    except Exception:
        logger.exception("Erreur lors de la génération de la réponse")
        return "Je suis désolé, j'ai rencontré un problème technique. Veuillez réessayer."
