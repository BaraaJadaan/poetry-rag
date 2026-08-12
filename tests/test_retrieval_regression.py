"""Fast, deterministic retrieval regression tests for CI."""

from __future__ import annotations

import numpy as np

import lancedb

from evaluate import evaluate_retriever
from retriever import HybridRetriever


def test_hybrid_retrieval_regression(tmp_path):
    rows = [
        {
            "id": "horse-and-night",
            "vector": [1.0, 0.0, 0.0],
            "text_index": "الخيل والليل والبيداء تعرفني",
            "text_display": "الخيل والليل والبيداء تعرفني",
        },
        {
            "id": "longing-and-separation",
            "vector": [0.0, 1.0, 0.0],
            "text_index": "الشوق والحنين والفراق",
            "text_display": "الشوق والحنين والفراق",
        },
        {
            "id": "wisdom-and-knowledge",
            "vector": [0.0, 0.0, 1.0],
            "text_index": "الحكمة والعلم نور يهدي",
            "text_display": "الحكمة والعلم نور يهدي",
        },
    ]
    query_vectors = {
        "الشجاعة وركوب الخيل": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "الحنين إلى الحبيب": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "طلب الحكمة": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    eval_set = [
        {"query": query, "expected_id": expected_id}
        for query, expected_id in [
            ("الشجاعة وركوب الخيل", "horse-and-night"),
            ("الحنين إلى الحبيب", "longing-and-separation"),
            ("طلب الحكمة", "wisdom-and-knowledge"),
        ]
    ]

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("fixture", data=rows)
    from lancedb.index import FTS
    table.create_index("text_index", config=FTS())

    retriever = HybridRetriever(
        lance_dir=str(tmp_path),
        table_name="fixture",
        query_embedder=lambda query: query_vectors[query],
    )
    metrics = evaluate_retriever(retriever, eval_set, top_k=3)

    assert metrics["Semantic"]["recall_at_1"] == 1.0
    assert metrics["Keyword"]["recall_at_5"] < 1.0
    assert metrics["Hybrid"]["recall_at_5"] == 1.0
    assert metrics["Hybrid"]["mrr"] == 1.0
