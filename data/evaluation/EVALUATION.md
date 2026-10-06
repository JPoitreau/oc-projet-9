# Évaluation Ragas — POC Puls-Events (Toulouse)

## 1. Contexte

Assistant RAG sur événements Open Agenda (Toulouse), index FAISS, génération OpenAI gpt-4o-mini.
L'énoncé OC mentionne Mistral ; ce POC utilise OpenAI pour embeddings et chat.

## 2. Protocole

- Dernière exécution : `2026-10-02T10:02:25.486669+00:00`
- Nombre de questions annotées : 12
- Jeu : `data/evaluation/annotated_qa.json`
- Commande : `uv run python src/evaluate_rag.py`

## 3. Résultats quantitatifs

| Métrique | Moyenne |
|----------|---------|
| faithfulness | 0.558 |
| answer_relevancy | 0.330 |
| context_precision | 0.556 |
| answer_correctness | 0.408 |

**Taux de réponses acceptables** (faithfulness et answer_relevancy ≥ 0.7) : **33.3%**

## 4. Interprétation des métriques

- **faithfulness** : mesure si la réponse invente des faits absents des contextes récupérés.
- **answer_relevancy** : mesure si la réponse traite bien la question posée.
- **context_precision** : mesure si les segments FAISS sont pertinents pour produire la réponse de référence.
- **answer_correctness** : proximité sémantique entre réponse générée et réponse annotée.

Les scores doivent être lus conjointement : un **faithfulness** élevé avec un **context_precision** faible indique des réponses prudentes mais mal alimentées ; un **answer_correctness** bas sur les questions `out_of_scope` peut refléter un refus formulé différemment de la référence humaine.

### Synthèse du run

- **Retrieval** (context_precision ≈ 0.56) : récupération partiellement alignée avec les réponses de référence.
- **Génération** (faithfulness ≈ 0.56, answer_relevancy ≈ 0.33) : qualité variable selon la formulation de la question.
- **Alignement référence** (answer_correctness ≈ 0.41) : écart sémantique modéré avec les réponses annotées.
- **Taux acceptable** : 33.3% des questions atteignent faithfulness et answer_relevancy ≥ 0.7.

## 5. Exemples qualitatifs

### Succès relatif
- **Q06** — Quand l'agence KIWI Indus/Logistique reçoit-elle à Borderouge ?

### Cas le plus faible (scores minimaux)
- **Q02** — Où se déroule le rendez-vous avec l'agence YES pour l'emploi industrie et logistique ?

## 6. Limites

- Indexation par description entière (chunking non activé dans rebuild).
- Coût API : génération + juge Ragas pour chaque question.
- Jeu annoté limité à Toulouse ; hors périmètre volontaire pour Q09–Q10.

## 7. Pistes

- Activer le chunking ; tester Mistral ; élargir le jeu annoté ; CI `workflow_dispatch`.
