# Image de base : Python 3.11 avec uv déjà installé
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Ajout des bibliothèques nécessaires pour LightGBM et PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# On n'installe pas les dépendances de développement
ENV UV_NO_DEV=1

# On copie le projet dans l'image
COPY . /app

# On installe les dépendances depuis uv.lock
RUN uv sync --locked

# On utilise l'environnement virtuel créé par uv
ENV PATH="/app/.venv/bin:$PATH"

# Imports app.* et utils.* (comme en local : uvicorn depuis src/)
ENV PYTHONPATH="/app/src"

# Port exposé (HF Spaces / run local -p 7860:7860)
EXPOSE 7860

# Lance le serveur FastAPI (api.py ne démarre pas uvicorn tout seul)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]