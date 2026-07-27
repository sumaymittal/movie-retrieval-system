from collections import defaultdict, Counter
import pickle
import os
import math
import statistics
from multiprocessing import Pool

from typing import Callable

from .search_utils import (
    BM25_B,
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    process_string,
    Movies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}

    def get_document(self, term: str) -> list[tuple[int, str]]:
        doc_ids: set[int] = self.index[term.lower()]

        docs: list[str] = []
        for id in doc_ids:
            docs.append(self.docmap[id])

        return sorted(list(zip(doc_ids, docs)))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = process_string()(term)
        if len(tokens) != 1:
            raise ValueError("invalid search term")
        token: str = tokens.pop()

        return self.term_frequencies[doc_id][token]

    def get_bm25_idf(self, term: str) -> float:
        tokens = process_string()(term)
        if len(tokens) != 1:
            raise ValueError("invalid search term")
        token: str = tokens.pop()

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))

        numerator = total_doc_count - term_match_doc_count + 0.5
        denominator = term_match_doc_count + 0.5

        return math.log(numerator / denominator + 1)

    def __get_average_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return statistics.mean(self.doc_lengths.values())

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        length_norm_factor = (
            1 - b + (b * (self.doc_lengths[doc_id] / self.__get_average_doc_length()))
        )

        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + (k1 * (length_norm_factor)))

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[tuple[int, str, float]]:
        processed_query = process_string()(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap.keys():
            scores[doc_id] = sum(self.bm25(doc_id, q) for q in processed_query)

        sorted_scores = sorted(scores, key=lambda s: scores[s], reverse=True)
        truncated = sorted_scores[:limit]

        results = [(s, self.docmap[s], scores[s]) for s in truncated]
        return results

    def build(self, movies: Movies) -> None:
        # build is CPU intensive. We can speed up process by using all cores
        num_workers = os.cpu_count() or 4
        chunk_size = math.ceil(len(movies) / num_workers)
        chunks = [movies[i : i + chunk_size] for i in range(0, len(movies), chunk_size)]

        with Pool(num_workers) as pool:
            partial_builds = pool.map(_build_partial_index, chunks)

        partial_indexes: list[dict[str, set[int]]] = []
        for (
            partial_index,
            partial_term_frequencies,
            partial_doc_lengths,
        ) in partial_builds:
            partial_indexes.append(partial_index)
            self.term_frequencies |= partial_term_frequencies
            self.doc_lengths |= partial_doc_lengths

        for pidx in partial_indexes:
            for token, ids in pidx.items():
                self.index[token] |= ids

        for movie in movies:
            self.docmap[movie["id"]] = movie["title"]

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

        with open(CACHE_DIR.joinpath("index.pkl"), "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(CACHE_DIR.joinpath("docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

        with open(CACHE_DIR.joinpath("term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(CACHE_DIR.joinpath("doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        index_path = CACHE_DIR.joinpath("index.pkl")
        docmap_path = CACHE_DIR.joinpath("docmap.pkl")
        term_frequencies_path = CACHE_DIR.joinpath("term_frequencies.pkl")
        doc_lengths_path = CACHE_DIR.joinpath("doc_lengths.pkl")

        if not any(
            [
                index_path.exists(),
                docmap_path.exists(),
                term_frequencies_path.exists(),
                doc_lengths_path.exists(),
            ]
        ):
            raise FileNotFoundError()

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def _build_partial_index(
    movies_chunk: Movies,
) -> tuple[dict[str, set[int]], dict[int, Counter[str]], dict[int, int]]:
    processor: Callable[[str], list[str]] = process_string()

    partial_index: dict[str, set[int]] = defaultdict(set)
    partial_tf: dict[int, Counter[str]] = {}
    partial_doc_lengths: dict[int, int] = {}

    for movie in movies_chunk:
        id = movie["id"]
        tokens = processor(movie["title"] + " " + movie["description"])
        for token in tokens:
            partial_index[token].add(id)

        partial_tf[id] = Counter(tokens)
        partial_doc_lengths[id] = len(tokens)

    return partial_index, partial_tf, partial_doc_lengths
