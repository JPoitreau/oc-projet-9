Template - Assistant intelligent de recommandation d’événements culturels
1. Objectifs du projet
●	Contexte : Présentez la mission confiée par Puls-Events.

●	Problématique : En quoi un système RAG répond-il aux besoins métier ?

●	Objectif du POC : Ce que vous cherchez à démontrer (faisabilité technique, valeur métier, performance).

●	Périmètre : Zone géographique ciblée, période d’événements, données utilisées.


2. Architecture du système
●	Schéma global (schéma UML) :

○	Données entrantes (API Open Agenda)

○	Prétraitement / embeddings / base vectorielle

○	Intégration LLM avec LangChain

○	Exposition via API

●	Technologies utilisées :

○	


3. Préparation et vectorisation des données
●	Source de données : API Open Agenda (paramètres utilisés, filtres appliqués)

●	Nettoyage : Exemples d’anomalies corrigées, méthodes utilisées

●	Chunking : Raison du découpage, taille choisie

●	Embedding :

○	Modèle utilisé (ex. : Mistral embedding API)

○	Dimensionnalité, logique de batch, format des vecteurs


4. Choix du modèle NLP
●	Modèle sélectionné :

●	Pourquoi ce modèle ? (Critères : coût, qualité, compatibilité LangChain…)

●	Prompting (si utilisé) : Prompt de base / structure

●	Limites du modèle : 

5. Construction de la base vectorielle
●	Faiss utilisé :

●	Stratégie de persistance : 
○	Format de sauvegarde ?
○	nommage ?

●	Métadonnées associées : 
○	Ce qui est conservé pour chaque document


6. API et endpoints exposés
●	Framework utilisé : FastAPI / Flask

●	Endpoints clés :

○	/ask : question utilisateur, réponse du système

○	/rebuild : reconstruction de l’index (si besoin)

●	Format des requêtes/réponses

●	Exemple d’appel API : avec curl ou code Python
●	Tests effectués et documentés

●	Gestion des erreurs / limitations


7. Évaluation du système
●	Jeu de test annoté :

○	**12 exemples** dans [`data/evaluation/annotated_qa.json`](data/evaluation/annotated_qa.json) (`question`, `reference_answer`, champs optionnels `id`, `category`, `notes`).
○	**Méthode d’annotation** : recherche manuelle dans `metadata.csv` ; réponse de référence rédigée à la main (2–5 phrases) ; catégories `factual`, `ambiguous`, `out_of_scope`, `retrieval_trap`.

●	Métriques d’évaluation (Ragas 0.3, script [`src/evaluate_rag.py`](src/evaluate_rag.py)) :

○	**faithfulness** — fidélité de la réponse aux contextes FAISS ;
○	**answer_relevancy** — pertinence par rapport à la question ;
○	**context_precision** — qualité du retrieval vs réponse de référence ;
○	**answer_correctness** — proximité avec la réponse annotée ;
○	**Taux acceptable** (maison) : part des questions avec faithfulness et answer_relevancy ≥ 0,7.

●	Résultats obtenus :

○	Fichiers générés sous `data/evaluation/` : `ragas_results.json`, `ragas_details.csv`, rapport [`EVALUATION.md`](data/evaluation/EVALUATION.md).
○	Analyse quantitative (moyennes Ragas) + qualitative (meilleur / pire cas dans le rapport).

●	**Commande** (clé `OPENAI_API_KEY` et index `data/processed/` requis) :

```bash
uv run python src/evaluate_rag.py
uv run python src/evaluate_rag.py --limit 3   # debug
uv run python src/evaluate_rag.py --from-cache   # Ragas seul (rag_predictions.json)
```

●	**pytest vs Ragas** : `pytest` valide le pipeline technique (mocks, sans API) ; Ragas mesure la qualité métier (coût et latence API). Une CI GitHub `workflow_dispatch` pourrait enchaîner les deux avec un secret `OPENAI_API_KEY`.

8. Recommandations et perspectives
●	Ce qui fonctionne bien

●	Limites du POC :

○	Volumétrie, performance, coût, couverture thématique ?

●	Améliorations possibles :

○	Ajout de…
Amélioration de..
○	Passage en production via…


9.  Organisation du dépôt GitHub
●	Arborescence du dépôt (fichiers clés, scripts, dossiers)

```
oc-projet-9/
├── data/
│   ├── raw/                    # CSV Open Agenda (gitignoré)
│   ├── processed/              # Index FAISS + metadata.csv
│   └── evaluation/             # annotated_qa.json, ragas_results.json, EVALUATION.md
├── src/
│   ├── app/                    # API FastAPI, Streamlit, chat, db_creation
│   ├── evaluate_rag.py         # Évaluation qualité Ragas
│   ├── utils/                  # Embeddings, recherche FAISS
│   └── notebooks/              # Exploration et prototypage
└── tests/
    ├── fixtures/               # Mini CSV pour tests reproductibles
    ├── unit/                   # Tests unitaires (nettoyage, index, prompt)
    ├── api_test.py             # Tests fonctionnels HTTP
    └── test_annotated_dataset.py # Schéma du jeu annoté (skip si absent)
```

●	Explication rapide de chaque répertoire

●	**Tests automatisés** (sans clé API : embeddings et LLM mockés) :

```bash
uv sync --group dev
uv run pytest
uv run pytest tests/api_test.py -v
uv run pytest -m slow   # perf légère + pipeline rebuild mini
```

Le fichier `tests/fixtures/sample_events.csv` sert de jeu minimal pour valider le nettoyage, l’indexation FAISS et l’API. Le jeu de référence humain pour l’évaluation qualitative peut être placé dans `data/evaluation/annotated_qa.json` (liste `{ "question", "reference_answer" }`).


10. Annexes (exemples)
●	Extraits du jeu de test annoté
●	Prompt utilisé (si spécifique)
●	Extraits de logs ou exemples de réponse JSON
Commandes d'import:
from mistralai.client import Mistral
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
