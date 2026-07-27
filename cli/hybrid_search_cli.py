from typing import cast
import argparse
import logging

from lib.hybrid_search import HybridSearch, normalize_scores, RRFSearchResult
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_K,
    DOC_PREVIEW_LENGTH,
    load_movies,
)
from lib.llm_utils import (
    get_gemini_client,
    query_gemini,
    get_spelling_query,
    get_rewritten_query,
    get_expanded_query,
    rerank_results_individual,
    rerank_results_batch,
    get_evaluation,
)
from lib.cross_encoder import rerank_cross_encoder

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    _ = normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="Scores to normalize using min-max normalization",
    )

    weighted_search_parser = subparsers.add_parser("weighted-search")
    _ = weighted_search_parser.add_argument("query", type=str, help="Search query")
    _ = weighted_search_parser.add_argument(
        "--alpha", type=float, help="Weight for BM25 scores (0.0 to 1.0)"
    )
    _ = weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of top results to return",
    )

    rrf_search_parser = subparsers.add_parser("rrf-search")
    _ = rrf_search_parser.add_argument("query", type=str, help="Search query")
    _ = rrf_search_parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help="Weight given to higher vs lower ranked scores",
    )
    _ = rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of top results to return",
    )
    _ = rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    _ = rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query rerank method",
    )
    _ = rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable LLM evaluation of results.",
    )

    return parser


def cmd_normalize(scores: list[float]) -> list[float]:
    return normalize_scores(scores)


def cmd_weighted_search(
    hybrid_search: HybridSearch, query: str, alpha: float, limit: int
):
    return hybrid_search.weighted_search(query, alpha, limit)


def cmd_rrf_search(
    hybrid_search: HybridSearch,
    query: str,
    k: float = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance_method: str = "",
    rerank_method: str = "",
) -> list[tuple[int, RRFSearchResult]]:
    """
    Perform a Reciprocal Rank Fusion (RRF) search with optional query enhancement and result reranking.

    This function executes a hybrid search using RRF, with optional preprocessing of the query
    through enhancement methods (spelling correction, rewriting, or expansion) and optional
    post-processing of results through various reranking strategies.

    Args:
        hybrid_search (HybridSearch): The hybrid search instance to perform the search.
        query (str): The search query string.
        k (float): The RRF constant parameter used in the reciprocal rank fusion calculation.
        limit (int): The maximum number of results to return.
        enhance_method (str, optional): Query enhancement method to apply. Valid options are:
            - "spell": Correct spelling errors in the query
            - "rewrite": Rewrite the query for better search performance
            - "expand": Expand the query with additional relevant terms
            - "" (empty): No enhancement (default)
        rerank_method (str, optional): Result reranking method to apply. Valid options are:
            - "individual": Rerank results individually using LLM evaluation
            - "batch": Rerank results in batch using LLM evaluation
            - "cross_encoder": Rerank using a cross-encoder model
            - "" (empty): No reranking (default)

    Returns:
        list[tuple[int, RRFSearchResult]]: A list of tuples containing the rank and search result,
            ordered by relevance. When reranking is applied, returns the top `limit` reranked results.
            Otherwise, returns the top `limit` results from the RRF search.

    Raises:
        ValueError: If an unrecognized enhance_method or rerank_method is provided.

    Notes:
        - When reranking is enabled, the function retrieves 5x the requested limit before reranking
          to ensure higher quality results after reranking.
        - Enhanced queries and reranking results are logged at debug level for monitoring.
        - Query enhancement prints the transformation to stdout for visibility.
    """
    gemini_client = get_gemini_client()

    if enhance_method:
        match enhance_method:
            case "spell":
                sys_prompt, contents = get_spelling_query(query)
                enhanced_query = query_gemini(gemini_client, sys_prompt, contents)
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case "rewrite":
                sys_prompt, contents = get_rewritten_query(query)
                enhanced_query = query_gemini(gemini_client, sys_prompt, contents)
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case "expand":
                sys_prompt, contents = get_expanded_query(query)
                enhanced_query = query_gemini(gemini_client, sys_prompt, contents)
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'"
                )
            case _:
                raise ValueError("unrecognised enhance method")
    else:
        enhanced_query = query

    logger.debug("Enhanced query: %s", enhanced_query)

    if rerank_method:
        higher_limit = limit * 5
        fast_results: list[tuple[int, RRFSearchResult]] = hybrid_search.rrf_search(
            enhanced_query, k, higher_limit
        )

        logger.debug(
            "Pre re-ranked results:\n%s",
            ", ".join(res[1]["title"] for res in fast_results),
        )

        match rerank_method:
            case "individual":
                reranked = rerank_results_individual(
                    gemini_client, enhanced_query, fast_results, limit
                )
            case "batch":
                reranked = rerank_results_batch(
                    gemini_client, enhanced_query, fast_results, limit
                )
            case "cross_encoder":
                reranked = rerank_cross_encoder(query, fast_results, limit)
            case _:
                raise ValueError("unrecognised rerank method")

        logger.debug(
            "Re-ranked results:\n%s",
            ", ".join(res[1]["title"] for res in reranked),
        )
        return reranked

    return hybrid_search.rrf_search(enhanced_query, k, limit)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    movie_data = load_movies()
    hybrid_search: HybridSearch = HybridSearch(movie_data)

    query: str = cast(str, args.query)
    limit: int = cast(int, args.limit)

    match args.command:
        case "normalize":
            scores: list[float] = cast(list[float], args.scores)
            normalized_scores: list[float] = cmd_normalize(scores)

            for s in normalized_scores:
                print(f"* {s:.4f}")

        case "weighted-search":
            alpha: float = cast(float, args.alpha)

            results = cmd_weighted_search(hybrid_search, query, alpha, limit)
            for i, (_, res) in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"     Hybrid Score: {res['hybrid']:.4f}")
                print(f"     BM25: {res['bm25']:.4f}, Semantic: {res['semantic']:.4f}")
                print(
                    f"     Description: {res['description'][:DOC_PREVIEW_LENGTH].replace('\n', ' ')}..."
                )

        case "rrf-search":
            k: float = cast(float, args.k)
            enhance_method: str = cast(str, args.enhance)
            rerank_method: str = cast(str, args.rerank_method)
            evaluate: bool = cast(bool, args.evaluate)

            logger.debug(
                "Running rrf-search cmd.\nQuery: %s\nk=%d\nEnhance Method: %s\nRerank Method: %s",
                query,
                k,
                enhance_method,
                rerank_method,
            )

            results = cmd_rrf_search(
                hybrid_search,
                query,
                k,
                limit,
                enhance_method,
                rerank_method,
            )
            for i, (_, res) in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                if res["rerank"] != -1:
                    print(f"     Rerank Score: {res['rerank']}")
                print(f"     RRF Score: {res['rrf']:.4f}")
                print(
                    f"     BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                )
                print(
                    f"     Description: {res['description'][:DOC_PREVIEW_LENGTH].replace('\n', ' ')}..."
                )

            if evaluate:
                gemini_client = get_gemini_client()
                get_evaluation(gemini_client, query, results)

        case _:
            parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
