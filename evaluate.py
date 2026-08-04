"""
Evaluation Harness for Phase 6.

Loads eval_set.json and tests all queries against Semantic, Keyword, and Hybrid
search strategies. Calculates Recall@1, Recall@5, and Mean Reciprocal Rank (MRR).
"""

import json
import sys
import time

try:
    from retriever import HybridRetriever
except ImportError:
    print("ERROR: retriever.py not found.")
    sys.exit(1)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def calculate_metrics(results_list, expected_id):
    """
    Returns (hit_at_1, hit_at_5, reciprocal_rank)
    results_list is a list of dictionaries/rows that have an 'id' field.
    """
    for rank, row in enumerate(results_list):
        if row["id"] == expected_id:
            return (1 if rank == 0 else 0, 1 if rank < 5 else 0, 1.0 / (rank + 1))
    return (0, 0, 0.0)

def main():
    try:
        with open("eval_set.json", "r", encoding="utf-8") as f:
            eval_set = json.load(f)
    except FileNotFoundError:
        print("ERROR: eval_set.json not found. Run build_golden_set.py first.")
        sys.exit(1)

    print(f"Loaded Golden Set with {len(eval_set)} queries.\n")
    retriever = HybridRetriever()

    metrics = {
        "Semantic": {"r1": 0, "r5": 0, "mrr": 0.0, "time": 0.0},
        "Keyword":  {"r1": 0, "r5": 0, "mrr": 0.0, "time": 0.0},
        "Hybrid":   {"r1": 0, "r5": 0, "mrr": 0.0, "time": 0.0},
    }

    n = len(eval_set)
    print(f"\nRunning evaluation on {n} queries...\n")

    for i, item in enumerate(eval_set):
        query = item["query"]
        expected_id = item["expected_id"]
        
        # 1. Semantic Search
        t0 = time.time()
        res_sem = retriever.search_semantic(query, limit=10).to_dict('records')
        t1 = time.time()
        r1, r5, mrr = calculate_metrics(res_sem, expected_id)
        metrics["Semantic"]["r1"] += r1
        metrics["Semantic"]["r5"] += r5
        metrics["Semantic"]["mrr"] += mrr
        metrics["Semantic"]["time"] += (t1 - t0)

        # 2. Keyword Search
        t0 = time.time()
        res_key = retriever.search_keyword(query, limit=10).to_dict('records')
        t1 = time.time()
        r1, r5, mrr = calculate_metrics(res_key, expected_id)
        metrics["Keyword"]["r1"] += r1
        metrics["Keyword"]["r5"] += r5
        metrics["Keyword"]["mrr"] += mrr
        metrics["Keyword"]["time"] += (t1 - t0)

        # 3. Hybrid Search
        t0 = time.time()
        res_hyb = retriever.search_hybrid(query, limit=10).to_dict('records')
        t1 = time.time()
        r1, r5, mrr = calculate_metrics(res_hyb, expected_id)
        metrics["Hybrid"]["r1"] += r1
        metrics["Hybrid"]["r5"] += r5
        metrics["Hybrid"]["mrr"] += mrr
        metrics["Hybrid"]["time"] += (t1 - t0)

        print(f"[{i+1}/{n}] Evaluated: '{query}'", flush=True)

    print("\n" + "="*60, flush=True)
    print("                    EVALUATION RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"{'Strategy':<12} | {'Recall@1':<10} | {'Recall@5':<10} | {'MRR':<10} | {'Avg Latency'}", flush=True)
    print("-" * 60, flush=True)
    
    for strategy, m in metrics.items():
        r1_pct = (m["r1"] / n) * 100
        r5_pct = (m["r5"] / n) * 100
        mrr_avg = m["mrr"] / n
        latency = (m["time"] / n) * 1000
        print(f"{strategy:<12} | {r1_pct:>8.1f}% | {r5_pct:>8.1f}% | {mrr_avg:>8.3f} | {latency:>7.1f} ms", flush=True)
    
    print("="*60, flush=True)
    
    print("\nConclusion: Hybrid search (RRF) should substantially outperform", flush=True)
    print("Keyword-only (since the queries use synonyms, not exact verse text)", flush=True)
    print("and should edge out Semantic-only on tricky edge cases.", flush=True)

if __name__ == "__main__":
    main()
