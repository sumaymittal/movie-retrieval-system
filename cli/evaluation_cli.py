"""
evaluation_cli.py
-----------------
Evaluate the Hoopla IR system against golden test cases using a comprehensive
set of IR metrics:

    Precision@k  |  Recall@k  |  F1@k  |  NDCG@k  |  MRR  |  MAP@k

Usage examples
--------------
# Basic RRF search, top-5
uv run python cli/evaluation_cli.py --limit 5

# BM25-only, top-10
uv run python cli/evaluation_cli.py --limit 10 --search-type bm25

# RRF with cross-encoder reranking
uv run python cli/evaluation_cli.py --limit 5 --rerank-method cross_encoder

# RRF with query expansion
uv run python cli/evaluation_cli.py --limit 5 --enhance expand
"""

import argparse
import json
import logging
from typing import cast

from hybrid_search_cli import cmd_rrf_search, HybridSearch
from lib.search_utils import load_movies, DEFAULT_K
from lib.eval_metrics import compute_all_metrics

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_DATASET_PATH = "./data/golden_dataset.json"

_COL_WIDTH = 10  # fixed column width for the summary table


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hoopla IR System — Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python cli/evaluation_cli.py --limit 5\n"
            "  uv run python cli/evaluation_cli.py --limit 10 --search-type bm25\n"
            "  uv run python cli/evaluation_cli.py --limit 5 --rerank-method cross_encoder\n"
            "  uv run python cli/evaluation_cli.py --limit 5 --enhance expand\n"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to retrieve and evaluate (k). Default: 5",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["rrf", "bm25", "semantic"],
        default="rrf",
        help="Retrieval method to evaluate. Default: rrf",
    )
    parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        default=None,
        help="Query enhancement to apply before search (RRF only). Default: none",
    )
    parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        default=None,
        help="Reranking method to apply after retrieval (RRF only). Default: none",
    )
    return parser


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_golden_dataset() -> dict:
    with open(DEFAULT_GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Retrieval dispatch
# ---------------------------------------------------------------------------

def retrieve(
    search_type: str,
    query: str,
    limit: int,
    enhance_method: str,
    rerank_method: str,
    hybrid_search: HybridSearch,
) -> list[str]:
    """Run the requested retrieval pipeline and return an ordered list of titles.

    For RRF without any LLM features, we call hybrid_search.rrf_search() directly
    to avoid the unconditional get_gemini_client() call inside cmd_rrf_search().
    """
    if search_type == "rrf":
        needs_llm = bool(enhance_method or rerank_method)

        if needs_llm:
            # cmd_rrf_search handles LLM enhancement + reranking but requires GEMINI_API_KEY
            results = cmd_rrf_search(
                hybrid_search,
                query,
                DEFAULT_K,
                limit,
                enhance_method=enhance_method,
                rerank_method=rerank_method,
            )
            return [res["title"] for _, res in results]
        else:
            # Plain RRF — no Gemini needed, call the underlying search directly
            results = hybrid_search.rrf_search(query, DEFAULT_K, limit)
            return [res["title"] for _, res in results]

    elif search_type == "bm25":
        results = hybrid_search._bm25_search(query, limit)
        return [title for _, title, _ in results]

    elif search_type == "semantic":
        semantic_results = hybrid_search.semantic_search.search_chunks(query, limit)
        sorted_results = sorted(
            semantic_results.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )
        return [res["title"] for _, res in sorted_results]

    else:
        raise ValueError(f"Unknown search type: {search_type}")


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

_METRICS = ["precision", "recall", "f1", "ndcg", "rr", "ap"]
_HEADERS = ["Precision", "Recall", "F1", "NDCG", "MRR", "MAP"]


def _header_row(k: int) -> str:
    query_col = f"{'Query':<40}"
    metrics_cols = "".join(f"{f'{h}@{k}' if h not in ('MRR',) else h:>{_COL_WIDTH}}" for h in _HEADERS)
    return query_col + metrics_cols


def _separator(k: int) -> str:
    return "-" * (40 + _COL_WIDTH * len(_METRICS))


def _data_row(query: str, metrics: dict[str, float]) -> str:
    truncated = (query[:37] + "...") if len(query) > 40 else query
    query_col = f"{truncated:<40}"
    metrics_cols = "".join(f"{metrics[m]:>{_COL_WIDTH}.4f}" for m in _METRICS)
    return query_col + metrics_cols


def _aggregate_row(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        m: sum(ms[m] for ms in all_metrics) / len(all_metrics)
        for m in _METRICS
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    k: int           = cast(int, args.limit)
    search_type: str = cast(str, args.search_type)
    enhance: str     = cast(str, args.enhance) if args.enhance else ""
    rerank: str      = cast(str, args.rerank_method) if args.rerank_method else ""

    print("\n" + "=" * 60)
    print("  Hoopla IR Evaluation")
    print("=" * 60)
    print(f"  Search type  : {search_type}")
    print(f"  k (cut-off)  : {k}")
    print(f"  Enhancement  : {enhance or 'none'}")
    print(f"  Reranking    : {rerank or 'none'}")
    print("=" * 60 + "\n")

    # Build search index once (warm-up)
    print("Loading movie data and building search indexes...")
    movie_data = load_movies()
    hybrid_search = HybridSearch(movie_data)
    print("Done.\n")

    golden_dataset = get_golden_dataset()
    test_cases = golden_dataset["test_cases"]

    all_metrics: list[dict[str, float]] = []
    per_query_results: list[tuple[str, list[str], set[str], dict[str, float]]] = []

    # ---- Per-query evaluation -------------------------------------------------
    print("Running evaluation across all test cases...\n")
    for tc in test_cases:
        query: str   = tc["query"]
        relevant: set[str] = set(tc["relevant_docs"])

        retrieved = retrieve(
            search_type=search_type,
            query=query,
            limit=k,
            enhance_method=enhance,
            rerank_method=rerank,
            hybrid_search=hybrid_search,
        )

        metrics = compute_all_metrics(retrieved, relevant, k)
        all_metrics.append(metrics)
        per_query_results.append((query, retrieved, relevant, metrics))

    # ---- Per-query detail block -----------------------------------------------
    for query, retrieved, relevant, metrics in per_query_results:
        hits = [t for t in retrieved if t in relevant]
        misses = [t for t in retrieved if t not in relevant]

        print(f"Query: \"{query}\"")
        print(f"  Retrieved  : {retrieved}")
        print(f"  Hits       : {hits}")
        print(f"  Misses     : {misses}")
        print(f"  Precision@{k}: {metrics['precision']:.4f}")
        print(f"  Recall@{k}   : {metrics['recall']:.4f}")
        print(f"  F1@{k}       : {metrics['f1']:.4f}")
        print(f"  NDCG@{k}     : {metrics['ndcg']:.4f}")
        print(f"  RR         : {metrics['rr']:.4f}")
        print(f"  AP@{k}       : {metrics['ap']:.4f}")
        print()

    # ---- Aggregate summary table ----------------------------------------------
    aggregate = _aggregate_row(all_metrics)

    sep = _separator(k)
    print(sep)
    print(f"  AGGREGATE SUMMARY  (macro-averaged over {len(test_cases)} queries, k={k})")
    print(sep)
    print(_header_row(k))
    print("-" * len(_header_row(k)))
    for query, _, _, metrics in per_query_results:
        print(_data_row(query, metrics))
    print("-" * len(_header_row(k)))
    print(_data_row("MACRO AVERAGE", aggregate))
    print(sep)
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
