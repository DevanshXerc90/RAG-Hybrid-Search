"""
Retrieval evaluation for the hybrid search pipeline.

Evaluates the retrieval modes actually used by PostgresSearcher against the
labelled query set in scripts/eval_queries.json and reports recall@5, MRR and
p50/p95 query latency for:

    - vector_only    : pgvector cosine distance ORDER BY ... LIMIT k
    - keyword_only   : PostgreSQL full-text search (ts_rank_cd)
    - hybrid_hnsw   : hybrid RRF fusion, vector branch served by the HNSW index
    - hybrid_no_hnsw : hybrid RRF fusion, vector branch served by a sequential
                       scan (the HNSW index is dropped for these measurements)

The HNSW index is toggled with DROP/CREATE INDEX around each measurement block,
and the index is restored afterwards so the application is unaffected.

Query embeddings (text-embedding-3-small) are computed once per query and
reused across modes and repeats, so the reported latency isolates the database
retrieval path. The embedding latency is reported separately.

Usage:
    python scripts/evaluate.py [--repeats N] [--json PATH]
"""

import argparse
import json
import math
import statistics
import sys
import time

sys.path.append(".")

from sqlalchemy import text  # noqa: E402

from models.database import get_db_session  # noqa: E402
from models.product import Product  # noqa: E402
from services.embedding import Embedding  # noqa: E402
from services.postgres_searcher import PostgresSearcher  # noqa: E402

QUERIES_PATH = "scripts/eval_queries.json"
TOP_K = 5
HNSW_INDEX_NAME = "eval_hnsw_embedding_on_embedding"
CREATE_HNSW_SQL = (
    f'CREATE INDEX {HNSW_INDEX_NAME} ON "Product" '
    "USING hnsw (embedding vector_cosine_ops)"
)


def drop_embedding_indexes() -> list[str]:
    """Drop every index on Product.embedding; returns the dropped names."""
    with get_db_session() as session:
        rows = session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'Product' AND indexdef ILIKE '%embedding%'"
            )
        ).fetchall()
        dropped = [row[0] for row in rows]
        for name in dropped:
            session.execute(text(f'DROP INDEX IF EXISTS "{name}"'))
        session.commit()
    return dropped


def ensure_hnsw_index() -> None:
    with get_db_session() as session:
        session.execute(text(f'DROP INDEX IF EXISTS "{HNSW_INDEX_NAME}"'))
        session.execute(text(CREATE_HNSW_SQL))
        session.commit()


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["queries"]


def recall_at_k(expected: list[int], retrieved: list[int]) -> float:
    if not expected:
        return 0.0
    return len(set(expected) & set(retrieved[:TOP_K])) / len(expected)


def reciprocal_rank(expected: list[int], retrieved: list[int]) -> float:
    expected_set = set(expected)
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in expected_set:
            return 1.0 / rank
    return 0.0


def percentile_nearest_rank(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(math.ceil(percentile / 100.0 * len(ordered)) - 1, 0)
    return ordered[index]


def evaluate_mode(
    mode: str,
    labelled_queries: list[dict],
    vectors: dict[str, list[float]],
    searcher: PostgresSearcher,
    repeats: int,
) -> dict:
    recalls, reciprocal_ranks, latencies_ms = [], [], []
    for item in labelled_queries:
        query_text = item["query"]
        expected = item["expected_ids"]
        query_vector = vectors.get(query_text, [])
        mode_recalls, mode_rrs = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            results = searcher.search(
                query_text if mode != "vector_only" else None,
                query_vector if mode != "keyword_only" else [],
                top=TOP_K,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            retrieved = [product.id for product in results if product is not None]
            latencies_ms.append(elapsed_ms)
            mode_recalls.append(recall_at_k(expected, retrieved))
            mode_rrs.append(reciprocal_rank(expected, retrieved))
        # Deterministic ranking metrics: identical across repeats.
        recalls.append(mode_recalls[0])
        reciprocal_ranks.append(mode_rrs[0])
    return {
        "mode": mode,
        "recall_at_5": statistics.mean(recalls),
        "mrr": statistics.mean(reciprocal_ranks),
        "p50_ms": percentile_nearest_rank(latencies_ms, 50),
        "p95_ms": percentile_nearest_rank(latencies_ms, 95),
        "samples": len(latencies_ms),
    }


def render_table(rows: list[dict]) -> str:
    header = (
        "| Mode | Recall@5 | MRR | p50 latency (ms) | p95 latency (ms) |"
    )
    separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['recall_at_5']:.3f} | {row['mrr']:.3f} "
            f"| {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timing repetitions per query per mode (default: 5)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="optional path to write raw results as JSON",
    )
    args = parser.parse_args()

    labelled_queries = load_queries()
    searcher = PostgresSearcher(Product)
    embedding_service = Embedding()

    with get_db_session() as session:
        product_count = session.execute(
            text('SELECT COUNT(*) FROM "Product"')
        ).scalar()
    if product_count == 0:
        raise SystemExit(
            "Product table is empty. Run `python scripts/load_data.py` first."
        )

    # Warm up the connection pool and measure query embedding latency once.
    vectors, embed_latencies_ms = {}, []
    for item in labelled_queries:
        start = time.perf_counter()
        vectors[item["query"]] = embedding_service.generate(item["query"])
        embed_latencies_ms.append((time.perf_counter() - start) * 1000.0)

    # One warm-up pass per mode so pool/planning costs are not timed.
    for mode in ("vector_only", "keyword_only", "hybrid_no_hnsw"):
        searcher.search(
            None if mode == "vector_only" else labelled_queries[0]["query"],
            [] if mode == "keyword_only" else vectors[labelled_queries[0]["query"]],
            top=TOP_K,
        )

    results = []
    try:
        # Hybrid without HNSW: vector branch falls back to a sequential scan.
        dropped = drop_embedding_indexes()
        print(f"[eval] dropped embedding indexes: {dropped or 'none'}")
        results.append(
            evaluate_mode(
                "hybrid_no_hnsw", labelled_queries, vectors, searcher, args.repeats
            )
        )

        # Hybrid with HNSW: rebuild a cosine-ops HNSW index for the vector branch.
        ensure_hnsw_index()
        print(f"[eval] created HNSW index {HNSW_INDEX_NAME}")
        results.append(
            evaluate_mode(
                "hybrid_hnsw", labelled_queries, vectors, searcher, args.repeats
            )
        )
        results.append(
            evaluate_mode(
                "vector_only", labelled_queries, vectors, searcher, args.repeats
            )
        )
        results.append(
            evaluate_mode(
                "keyword_only", labelled_queries, vectors, searcher, args.repeats
            )
        )
    finally:
        # Leave a working HNSW index in place for the application.
        ensure_hnsw_index()

    mode_order = ["vector_only", "keyword_only", "hybrid_hnsw", "hybrid_no_hnsw"]
    results.sort(key=lambda row: mode_order.index(row["mode"]))

    print(f"\n[eval] products in table: {product_count}")
    print(f"[eval] labelled queries: {len(labelled_queries)}")
    print(f"[eval] timing repeats per query per mode: {args.repeats}")
    print(
        "[eval] query embedding latency p50/p95 (ms): "
        f"{percentile_nearest_rank(embed_latencies_ms, 50):.2f} / "
        f"{percentile_nearest_rank(embed_latencies_ms, 95):.2f}"
        " (excluded from per-mode latency)"
    )
    print()
    print(render_table(results))

    if args.json_path:
        payload = {
            "products": product_count,
            "queries": len(labelled_queries),
            "repeats": args.repeats,
            "embedding_p50_ms": percentile_nearest_rank(embed_latencies_ms, 50),
            "embedding_p95_ms": percentile_nearest_rank(embed_latencies_ms, 95),
            "results": results,
        }
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n[eval] wrote raw results to {args.json_path}")


if __name__ == "__main__":
    main()
