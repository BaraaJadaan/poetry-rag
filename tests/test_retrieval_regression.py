"""Fast, deterministic retrieval regression tests for CI."""

from __future__ import annotations

import numpy as np

import lancedb

from evaluate import evaluate_retriever
from retriever import HybridRetriever, is_full_verse


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


def test_full_verse_python_filter(tmp_path):
    """Fragment-marker rows (with trailing whitespace that defeated SQL LIKE)
    must be excluded by the Python post-filter on both search paths."""
    rows = [
        {
            "id": "full1",
            "vector": [1.0, 0.0, 0.0],
            "text_index": "وليت الذي بيني وبينك عامر وبينك غور فالبلاد البعيد",
            "text_display": "وليت الذي بيني وبينك عامر *** وبينك غور فالبلاد البعيد",
        },
        {
            "id": "orphan1",
            "vector": [0.95, 0.0, 0.05],
            "text_index": "وليت الذي بيني وبينك عامر",
            "text_display": "وليت الذي بيني وبينك عامر",
        },
        {
            "id": "frag_end",
            "vector": [0.9, 0.0, 0.1],
            "text_index": "والعيون ***",
            "text_display": "والعيون ***\n",
        },
        {
            "id": "frag_start",
            "vector": [0.8, 0.1, 0.0],
            "text_index": "*** فيرجع الصدى",
            "text_display": "*** فيرجع الصدى\n",
        },
    ]
    query_vectors = {
        "وليت الذي بيني وبينك عامر": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "والعيون ***": np.array([0.9, 0.0, 0.1], dtype=np.float32),
    }

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("fixture", data=rows)
    from lancedb.index import FTS

    table.create_index("text_index", config=FTS())

    retriever = HybridRetriever(
        lance_dir=str(tmp_path),
        table_name="fixture",
        query_embedder=lambda query: query_vectors[query],
    )

    df = retriever.search_hybrid("والعيون ***", limit=10, filter_fn=is_full_verse)
    ids = set(df["id"])
    assert "full1" in ids
    assert "orphan1" not in ids
    assert "frag_end" not in ids
    assert "frag_start" not in ids

    df_sem = retriever.search_semantic("والعيون ***", limit=10, filter_fn=is_full_verse)
    assert "frag_end" not in set(df_sem["id"])
    assert "frag_start" not in set(df_sem["id"])
