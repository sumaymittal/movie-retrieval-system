from typing import TypedDict
import logging

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch, SemanticResult
from .search_utils import Movies, Movie, DEFAULT_SEARCH_LIMIT


logger = logging.getLogger(__name__)


class HybridSearchResult(TypedDict):
    title: str
    description: str
    bm25: float
    semantic: float
    hybrid: float


class RRFSearchResult(TypedDict):
    title: str
    description: str
    bm25_rank: int
    semantic_rank: int
    rrf: float
    rerank: int


type RRFSearchResults = list[tuple[int, RRFSearchResult]]


class HybridSearch:
    def __init__(self, documents: Movies) -> None:
        self.docmap: dict[int, Movie] = {}
        for doc in documents:
            self.docmap[doc["id"]] = doc

        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        _ = self.semantic_search.load_or_create_embeddings(documents)

        self.idx: InvertedIndex = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build(documents)
            self.idx.save()
        except Exception as e:
            print("Error loading keyword index: ", e)
            raise e

    def _bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[tuple[int, str, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        logger.info("starting weighted search")

        bm25_results: list[tuple[int, str, float]] = self._bm25_search(
            query, limit * 500
        )
        normalized_bm25_scores: list[float] = normalize_scores(
            [score for _, _, score in bm25_results]
        )
        normalized_bm25_results: dict[int, float] = {
            id: score for (id, _, _), score in zip(bm25_results, normalized_bm25_scores)
        }
        logger.info("bm25 results retrieved and normalized")

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        normalized_semantic_scores: list[float] = normalize_scores(
            [res["score"] for res in semantic_results.values()]
        )
        normalized_semantic_results: dict[int, float] = {
            int(id): score
            for id, score in zip(semantic_results.keys(), normalized_semantic_scores)
        }
        logger.info("semantic results retrieved and normalized")

        combined_results: dict[int, HybridSearchResult] = {}
        for id in set(normalized_bm25_results) | set(normalized_semantic_results):
            bm25 = normalized_bm25_results.get(id, 0.0)
            semantic = normalized_semantic_results.get(id, 0.0)

            combined_results[id] = HybridSearchResult(
                title=self.docmap[id]["title"],
                description=self.docmap[id]["description"],
                bm25=bm25,
                semantic=semantic,
                hybrid=_weighted_score(bm25, semantic, alpha),
            )

        logger.info("results combined and weighted scores assigned. unsorted")
        return sorted(
            combined_results.items(),
            key=lambda item: item[1]["hybrid"],
            reverse=True,
        )[:limit]

    def rrf_search(self, query, k, limit=10) -> RRFSearchResults:
        bm25_mapped: dict[int, float] = {
            id: score for id, _, score in self._bm25_search(query, limit * 500)
        }
        bm25_sorted: list[tuple[int, float]] = sorted(
            bm25_mapped.items(), key=lambda item: item[1], reverse=True
        )
        logger.info("bm25 results retrieved and normalized")

        semantic_mapped: dict[int, SemanticResult] = self.semantic_search.search_chunks(
            query, limit * 500
        )
        semantic_sorted: list[tuple[int, SemanticResult]] = sorted(
            semantic_mapped.items(), key=lambda item: item[1]["score"], reverse=True
        )
        logger.info("semantic results retrieved and normalized")

        combined_results: dict[int, RRFSearchResult] = {}

        rank = 1
        for id, _ in bm25_sorted:
            combined_results.setdefault(
                id,
                RRFSearchResult(
                    title=self.docmap[id]["title"],
                    description=self.docmap[id]["description"],
                    bm25_rank=0,
                    semantic_rank=0,
                    rrf=0.0,
                    rerank=-1,
                ),
            )["bm25_rank"] = rank
            rank += 1

        rank = 1
        for id, _ in semantic_sorted:
            combined_results.setdefault(
                id,
                RRFSearchResult(
                    title=self.docmap[id]["title"],
                    description=self.docmap[id]["description"],
                    bm25_rank=0,
                    semantic_rank=0,
                    rrf=0.0,
                    rerank=-1,
                ),
            )["semantic_rank"] = rank
            rank += 1

        for id in combined_results:
            bm25_rrf, semantic_rrf = 0, 0

            if combined_results[id]["bm25_rank"] != 0:
                bm25_rrf = _rrf_score(combined_results[id]["bm25_rank"], k)

            if combined_results[id]["semantic_rank"] != 0:
                semantic_rrf = _rrf_score(combined_results[id]["semantic_rank"], k)

            combined_results[id]["rrf"] = bm25_rrf + semantic_rrf

        logger.info("results combined and weighted scores assigned. unsorted")
        return sorted(
            combined_results.items(),
            key=lambda item: item[1]["rrf"],
            reverse=True,
        )[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(s - min_score) / (max_score - min_score) for s in scores]


def _weighted_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def _rrf_score(rank, k=60):
    return 1 / (k + rank)
