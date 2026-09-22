from openai import OpenAI
import numpy as np
import faiss
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_embeddings(text, model="text-embedding-3-small"):
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def search_relevant_segments(raw,index,question, k=5):
    """
    Recherche les segments les plus pertinents par rapport à la question posée.
    Args:
        question (str): La question posée par l'utilisateur
        k (int): Nombre de segments à récupérer
    Returns:
        list: Liste des segments pertinents
    """
    # Obtenir l'embedding de la question
    question_embedding = get_embeddings(question)
    question_embedding = np.array([question_embedding]).astype('float32')
    # Recherche dans l'index
    _ , indices = index.search(question_embedding, k) # _ is the distance
    segments = []
    for idx in indices[0]:
        row = raw.loc[raw.uid == idx].iloc[0] 
        segments.append(format_row_as_key_value(row))
    return segments 

def load_index_and_raw():
    index_path = DATA_DIR / "faiss_index.idx"
    raw_path = DATA_DIR / "metadata.csv"
    index = faiss.read_index(str(index_path))
    raw = pd.read_csv(raw_path)
    return index, raw

def format_row_as_key_value(row: pd.Series, skip_empty: bool = True) -> str:
    """Convertit une ligne du DataFrame en texte clé-valeur."""
    lines = []
    for col, val in row.items():
        if skip_empty and (pd.isna(val) or str(val).strip() == ""):
            continue
        lines.append(f"{col}: {val}")
    return "\n".join(lines)