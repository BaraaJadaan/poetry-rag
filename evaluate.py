"""Evaluation harness for semantic, keyword, and hybrid retrieval.

The evaluation functions are intentionally reusable by CI fixtures. The command-line
entrypoint can optionally wrap a full evaluation in an MLflow run without making MLflow
part of the serving container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def calculate_metrics(results_list: list[dict[str, Any]], expected_id: str) -> tuple[int, int, float]:
    """Return hit@1, hit@5, and reciprocal rank for one query."""
    for rank, row in enumerate(results_list):
        if row["id"] == expected_id:
            return (1 if rank == 0 else 0, 1 if rank < 5 else 0, 1.0 / (rank + 1))
    return (0, 0, 0.0)


def evaluate_retriever(retriever, eval_set: list[dict[str, Any]], top_k: int = 10) -> dict[str, dict[str, float]]:
    """Run the three retrieval strategies and return normalized metrics."""
    metrics: dict[str, dict[str, float]] = {
        "Semantic": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "latency_ms": 0.0},
        "Keyword": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "latency_ms": 0.0},
        "Hybrid": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "latency_ms": 0.0},
    }

    for item in eval_set:
        query = item["query"]
        expected_id = item["expected_id"]
        searches = {
            "Semantic": retriever.search_semantic,
            "Keyword": retriever.search_keyword,
            "Hybrid": retriever.search_hybrid,
        }

        for strategy, search in searches.items():
            started = time.perf_counter()
            results = search(query, limit=top_k).to_dict("records")
            elapsed_ms = (time.perf_counter() - started) * 1000
            r1, r5, mrr = calculate_metrics(results, expected_id)
            metrics[strategy]["recall_at_1"] += r1
            metrics[strategy]["recall_at_5"] += r5
            metrics[strategy]["mrr"] += mrr
            metrics[strategy]["latency_ms"] += elapsed_ms

    count = len(eval_set)
    if count == 0:
        raise ValueError("Evaluation set must contain at least one query.")

    for values in metrics.values():
        values["recall_at_1"] /= count
        values["recall_at_5"] /= count
        values["mrr"] /= count
        values["latency_ms"] /= count

    return metrics


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Evaluation set must be a JSON list: {path}")
    return data


def print_results(metrics: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print("                           EVALUATION RESULTS")
    print("=" * 72)
    print(f"{'Strategy':<12} | {'Recall@1':<10} | {'Recall@5':<10} | {'MRR':<10} | {'Avg latency'}")
    print("-" * 72)
    for strategy, values in metrics.items():
        print(
            f"{strategy:<12} | {values['recall_at_1'] * 100:>8.1f}% | "
            f"{values['recall_at_5'] * 100:>8.1f}% | {values['mrr']:>8.3f} | "
            f"{values['latency_ms']:>8.1f} ms"
        )
    print("=" * 72)


def log_mlflow_run(metrics: dict[str, dict[str, float]], eval_count: int, top_k: int,
                   table_name: str, table_size: int) -> None:
    """Log one evaluation to local or configured MLflow tracking storage."""
    import mlflow

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "poetry-rag-retrieval")
    mlflow.set_experiment(experiment_name)
    run_name = os.getenv("MLFLOW_RUN_NAME")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "git_commit": os.getenv("GITHUB_SHA", "local"),
            "corpus_version": os.getenv("CORPUS_VERSION", "unspecified"),
        })
        mlflow.log_params({
            "embedding_mode": "openrouter" if os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true" else "local",
            "embedding_model": "qwen/qwen3-embedding-8b" if os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true" else "voyage-4-nano",
            "table_name": table_name,
            "table_size": table_size,
            "evaluation_queries": eval_count,
            "top_k": top_k,
        })

        for strategy, values in metrics.items():
            prefix = strategy.lower()
            mlflow.log_metrics({
                f"{prefix}_recall_at_1": values["recall_at_1"],
                f"{prefix}_recall_at_5": values["recall_at_5"],
                f"{prefix}_mrr": values["mrr"],
                f"{prefix}_latency_ms": values["latency_ms"],
            })

        mlflow.log_dict(metrics, "retrieval_metrics.json")
        print(f"MLflow run logged in experiment '{experiment_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Poetry RAG retriever.")
    parser.add_argument("--eval-set", default="eval_set.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mlflow", action="store_true", help="Track this evaluation in MLflow.")
    parser.add_argument("--output", help="Optional path for a JSON metrics report.")
    args = parser.parse_args()

    try:
        from retriever import HybridRetriever
        eval_set = load_eval_set(args.eval_set)
        retriever = HybridRetriever()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Loaded {len(eval_set)} evaluation queries.")
    metrics = evaluate_retriever(retriever, eval_set, top_k=args.top_k)
    print_results(metrics)

    report = {
        "evaluation_set": str(args.eval_set),
        "query_count": len(eval_set),
        "top_k": args.top_k,
        "metrics": metrics,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    if args.mlflow:
        try:
            log_mlflow_run(metrics, len(eval_set), args.top_k, retriever.table_name, len(retriever.tbl))
        except ImportError:
            print("ERROR: MLflow is not installed. Run: uv sync --group mlops", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
