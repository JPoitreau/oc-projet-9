
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.utils import PROJECT_ROOT, RAW_DATA_PATH, get_embeddings


def check_for_dry_emails_urls_or_numeric(text_input):
    URL_PATTERN = re.compile(r'^https?://\S+$', re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

    text = text_input.strip()

    if URL_PATTERN.fullmatch(text):
        return {"valid": False, "reason": "url_only"}

    if EMAIL_PATTERN.fullmatch(text):
        return {"valid": False, "reason": "email_only"}

    if text.isnumeric():
        return {"valid": False, "reason": "numeric_only"}

    return {
        "valid": True,
        "reason": None
    }

def normalize_text(text):
    text = text.lower() # Convertit tous les caractères en minuscules.
    text = re.sub(r'\s+', ' ', text) # Remplace les espaces multiples par un seul espace.
    text = re.sub(r"(?<=[a-zà-ÿ])-(?=[a-zà-ÿ])", " ", text) # Remplace les tirets par des espaces
    text = re.sub(r"[^a-z0-9à-ÿ\s']", "", text) # Supprime tous les caractères non alphanumériques.
    if re.match(r"^de\s+\w+$", text):
        text = re.sub(r"^de\s+", "", text) # Supprime le mot "de" au début de la description si présent (ex "de piano")
    return text.strip() # Supprime les espaces en trop au début et à la fin.

def clean_text(text):
    quality = check_for_dry_emails_urls_or_numeric(text)
    if quality["valid"]:
        text = normalize_text(text)
        return text
    else:
        return None

def split_text_into_chunks(text, chunk_size = 1000, chunk_overlap = 200):
    # Initialisation du splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Taille de chaque segment
        chunk_overlap=chunk_overlap # Chevauchement entre les segments
        )
    # Découpage du texte
    segments = text_splitter.split_text(text)
    return segments

#def get_embeddings(text, model="text-embedding-3-small"):
#    client = OpenAI()
#    response = client.embeddings.create(
#        input=text,
#        model=model
#    )
#    return response.data[0].embedding

def rebuild_faiss_index():
    """
    Rebuild the Faiss index and save it to the disk.
    Args:
        None
    Returns:
        dict: Information about the rebuilt index and metadata.
    """

    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
    INDEX_PATH = OUTPUT_PATH / "faiss_index.idx"
    METADATA_PATH = OUTPUT_PATH / "metadata.csv"

    # Read the raw data.
    try:
        raw = pd.read_csv(RAW_DATA_PATH, sep=";")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {RAW_DATA_PATH} does not exist.")
    
    # Create the output path if it doesn't exist.
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Raw data loaded successfully. Starting cleaning...")
    # Filter out rows with unusable descriptions.
    clean_description = (
        raw['description_fr']
        .dropna()
        .apply(clean_text)
        .dropna()
    )

    raw = raw[raw.index.isin(clean_description.index)]

    # Define documents and ids for the Faiss index.
    documents = clean_description.tolist()
    ids = np.array(raw.uid.tolist())

    # Generate embeddings for each document.
    print("Generating embeddings...")
    embeddings = np.array([get_embeddings(doc) for doc in documents])
    print("Embeddings generated successfully. Initializing Faiss index...")
    # Initialize the Faiss index with explicit IDs.
    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

    # Add embeddings to the index.
    index.add_with_ids(embeddings, ids)
    print("Faiss index initialized successfully. Saving index and metadata...")
    # Save the index and metadata to the disk.
    faiss.write_index(index, str(INDEX_PATH))
    raw.to_csv(METADATA_PATH)

    return {
        "nb of indexed documents": len(documents),
        "dimension of the embeddings": dimension,
        "index path": str(INDEX_PATH),
        "metadata path": str(METADATA_PATH)
    }

def _demo_rebuild_and_search():  # pragma: no cover
    load_dotenv()

    root = Path(__file__).resolve().parent
    if root.name == "notebooks" and root.parent.name == "src":
        root = root.parent.parent

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output_path = root / "data" / "processed"
    index_path = output_path / "faiss_index.idx"
    metadata_path = output_path / "metadata.csv"

    rebuild_faiss_index()

    index = faiss.read_index(index_path)
    raw = pd.read_csv(metadata_path)
    query = "Informations sur les cours de piano"
    query_embedding = get_embeddings(query)
    k = 5
    distances, indices = index.search(np.array([query_embedding]), k)

    for rank, idx in enumerate(indices[0]):
        print(
            f"Document {idx}: {raw.loc[raw.uid == idx, 'description_fr']} - Distance: {distances[0][rank]}"
        )


if __name__ == "__main__":
    _demo_rebuild_and_search()
