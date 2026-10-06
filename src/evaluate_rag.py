"""
Évaluation qualité du RAG avec Ragas sur le jeu annoté.
Usage (depuis la racine du repo) :
    uv run python src/evaluate_rag.py
    uv run python src/evaluate_rag.py --limit 3
    uv run python src/evaluate_rag.py --from-cache   # après interruption
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    faithfulness,
)

from app.chat import run_rag_query
from utils.utils import PROJECT_ROOT, load_index_and_raw

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
ANNOTATED_PATH = EVAL_DIR / "annotated_qa.json"
RESULTS_JSON = EVAL_DIR / "ragas_results.json"
DETAILS_CSV = EVAL_DIR / "ragas_details.csv"
SUMMARY_MD = EVAL_DIR / "EVALUATION.md"
PREDICTIONS_CACHE = EVAL_DIR / "rag_predictions.json"

THRESHOLD = 0.7
METRICS = [faithfulness, answer_relevancy, context_precision, answer_correctness]
METRIC_NAMES = [m.name for m in METRICS]


def load_annotated(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Le jeu annoté doit être une liste non vide.")
    return data


def build_evaluation_dataset(items: list[dict], raw, index) -> tuple[Dataset, list[dict]]:
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    rows_meta = []

    for i, item in enumerate(items):
        qid = item.get("id", f"Q{i + 1}")
        question = item["question"]
        print(f"[{qid}] Génération RAG… {question[:60]}…")
        answer, ctx_list = run_rag_query(question, raw, index)
        print(f"[{qid}] OK ({len(ctx_list)} segments)")

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx_list)
        ground_truths.append(item["reference_answer"])
        rows_meta.append(
            {
                "id": qid,
                "category": item.get("category", ""),
                "question": question,
                "reference_answer": item["reference_answer"],
                "generated_answer": answer,
                "context_count": len(ctx_list),
            }
        )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    return dataset, rows_meta


def save_predictions_cache(dataset: Dataset, rows_meta: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": list(dataset["question"]),
        "answers": list(dataset["answer"]),
        "contexts": list(dataset["contexts"]),
        "ground_truths": list(dataset["ground_truth"]),
        "rows_meta": rows_meta,
    }
    PREDICTIONS_CACHE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_predictions_cache() -> tuple[Dataset, list[dict]]:
    payload = json.loads(PREDICTIONS_CACHE.read_text(encoding="utf-8"))
    dataset = Dataset.from_dict(
        {
            "question": payload["questions"],
            "answer": payload["answers"],
            "contexts": payload["contexts"],
            "ground_truth": payload["ground_truths"],
        }
    )
    return dataset, payload["rows_meta"]


def run_ragas_evaluation(dataset: Dataset):
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return evaluate(
        dataset,
        metrics=METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )


def write_outputs(result, rows_meta: list[dict]):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df_scores = result.to_pandas()

    details = pd.DataFrame(rows_meta)
    for name in METRIC_NAMES:
        if name in df_scores.columns:
            details[name] = df_scores[name].values

    details.to_csv(DETAILS_CSV, index=False, encoding="utf-8")

    metric_means = {
        name: float(df_scores[name].mean(skipna=True))
        for name in METRIC_NAMES
        if name in df_scores.columns
    }
    if "faithfulness" in df_scores.columns and "answer_relevancy" in df_scores.columns:
        acceptable_rate = float(
            (
                (df_scores["faithfulness"] >= THRESHOLD)
                & (df_scores["answer_relevancy"] >= THRESHOLD)
            ).mean()
        )
    else:
        acceptable_rate = 0.0

    score_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    if score_cols:
        worst_idx = int(df_scores[score_cols].min(axis=1).idxmin())
    else:
        worst_idx = 0

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows_meta),
        "metrics_mean": metric_means,
        "acceptable_rate": acceptable_rate,
        "threshold": THRESHOLD,
        "worst_question_index": worst_idx,
        "worst_question_id": rows_meta[worst_idx].get("id"),
    }
    RESULTS_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Résumé Ragas ---")
    for name, value in metric_means.items():
        print(f"  {name}: {value:.3f}")
    print(f"  taux acceptable (faithfulness & answer_relevancy >= {THRESHOLD}): {acceptable_rate:.1%}")
    print(f"  pire cas: {rows_meta[worst_idx].get('id')} — {rows_meta[worst_idx]['question'][:80]}…")
    print(f"\nFichiers écrits:\n  {RESULTS_JSON}\n  {DETAILS_CSV}")

    update_evaluation_md(summary, details, worst_idx)
    return summary


def update_evaluation_md(summary: dict, details: pd.DataFrame, worst_idx: int):
    """Met à jour les sections quantitatives de EVALUATION.md (squelette + chiffres du run)."""
    means = summary.get("metrics_mean", {})
    lines = [
        "# Évaluation Ragas — POC Puls-Events (Toulouse)",
        "",
        "## 1. Contexte",
        "",
        "Assistant RAG sur événements Open Agenda (Toulouse), index FAISS, génération OpenAI gpt-4o-mini.",
        "L'énoncé OC mentionne Mistral ; ce POC utilise OpenAI pour embeddings et chat.",
        "",
        "## 2. Protocole",
        "",
        f"- Dernière exécution : `{summary.get('run_at', 'N/A')}`",
        f"- Nombre de questions annotées : {summary.get('n_questions')}",
        "- Jeu : `data/evaluation/annotated_qa.json`",
        "- Commande : `uv run python src/evaluate_rag.py`",
        "",
        "## 3. Résultats quantitatifs",
        "",
        "| Métrique | Moyenne |",
        "|----------|---------|",
    ]
    for name in METRIC_NAMES:
        val = means.get(name)
        lines.append(f"| {name} | {val:.3f} |" if val is not None else f"| {name} | — |")
    lines.extend(
        [
            "",
            f"**Taux de réponses acceptables** (faithfulness et answer_relevancy ≥ {THRESHOLD}) : "
            f"**{summary.get('acceptable_rate', 0):.1%}**",
            "",
            "## 4. Interprétation des métriques",
            "",
            "- **faithfulness** : mesure si la réponse invente des faits absents des contextes récupérés.",
            "- **answer_relevancy** : mesure si la réponse traite bien la question posée.",
            "- **context_precision** : mesure si les segments FAISS sont pertinents pour produire la réponse de référence.",
            "- **answer_correctness** : proximité sémantique entre réponse générée et réponse annotée.",
            "",
            "Les scores doivent être lus conjointement : un **faithfulness** élevé avec un "
            "**context_precision** faible indique des réponses prudentes mais mal alimentées ; un "
            "**answer_correctness** bas sur les questions `out_of_scope` peut refléter un refus "
            "formulé différemment de la référence humaine.",
            "",
            "### Synthèse du run",
            "",
            f"- **Retrieval** (context_precision ≈ {means.get('context_precision', 0):.2f}) : "
            "récupération partiellement alignée avec les réponses de référence.",
            f"- **Génération** (faithfulness ≈ {means.get('faithfulness', 0):.2f}, "
            f"answer_relevancy ≈ {means.get('answer_relevancy', 0):.2f}) : "
            "qualité variable selon la formulation de la question.",
            f"- **Alignement référence** (answer_correctness ≈ {means.get('answer_correctness', 0):.2f}) : "
            "écart sémantique modéré avec les réponses annotées.",
            f"- **Taux acceptable** : {summary.get('acceptable_rate', 0):.1%} des questions "
            f"atteignent faithfulness et answer_relevancy ≥ {THRESHOLD}.",
            "",
            "## 5. Exemples qualitatifs",
            "",
        ]
    )
    if not details.empty:
        score_cols = [c for c in METRIC_NAMES if c in details.columns]
        if score_cols:
            best_idx = int(details[score_cols].mean(axis=1).idxmax())
            row_b = details.iloc[best_idx]
            row_w = details.iloc[worst_idx]
            lines.extend(
                [
                    "### Succès relatif",
                    f"- **{row_b.get('id', '')}** — {row_b['question']}",
                    "",
                    "### Cas le plus faible (scores minimaux)",
                    f"- **{row_w.get('id', '')}** — {row_w['question']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 6. Limites",
            "",
            "- Indexation par description entière (chunking non activé dans rebuild).",
            "- Coût API : génération + juge Ragas pour chaque question.",
            "- Jeu annoté limité à Toulouse ; hors périmètre volontaire pour Q09–Q10.",
            "",
            "## 7. Pistes",
            "",
            "- Activer le chunking ; tester Mistral ; élargir le jeu annoté ; CI `workflow_dispatch`.",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {SUMMARY_MD}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation Ragas du POC RAG")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de questions (debug)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Réutiliser rag_predictions.json (saute la génération RAG)",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY manquante (génération RAG + métriques Ragas).", file=sys.stderr)
        sys.exit(1)

    if not ANNOTATED_PATH.exists():
        print(f"Jeu annoté introuvable : {ANNOTATED_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        index, raw = load_index_and_raw()
    except FileNotFoundError as exc:
        print(f"{exc}\nLancez db_creation ou GET /rebuild/ avant l'évaluation.", file=sys.stderr)
        sys.exit(1)

    if args.from_cache:
        if not PREDICTIONS_CACHE.exists():
            print(f"Cache introuvable : {PREDICTIONS_CACHE}", file=sys.stderr)
            sys.exit(1)
        print(f"Chargement du cache {PREDICTIONS_CACHE}")
        dataset, rows_meta = load_predictions_cache()
    else:
        items = load_annotated(ANNOTATED_PATH)
        if args.limit:
            items = items[: args.limit]
        dataset, rows_meta = build_evaluation_dataset(items, raw, index)
        save_predictions_cache(dataset, rows_meta)
        print(f"Prédictions RAG enregistrées dans {PREDICTIONS_CACHE}")

    result = run_ragas_evaluation(dataset)
    write_outputs(result, rows_meta)


if __name__ == "__main__":
    main()
