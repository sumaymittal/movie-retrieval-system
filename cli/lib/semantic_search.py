from collections import defaultdict
import json

from typing import Any, override, TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    DOC_PREVIEW_LENGTH,
    SCORE_PRECISION,
    Movie,
    Movies,
    SimilarityScore,
    cosine_similarity,
    semantic_chunking,
)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticResult(TypedDict):
    title: str
    document: str
    score: float
    metadata: Any


def _format_search_result(
    title: str, document: str, score: float, **metadata: Any
) -> SemanticResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding: np.ndarray = self.model.encode([text])
        return embedding[0]

    def _build_embeddings(self, documents: Movies):
        movie_strings: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        np.save(CACHE_DIR.joinpath("movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: Movies):
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_path = CACHE_DIR.joinpath("movie_embeddings.npy")
        if not embeddings_path.exists():
            return self._build_embeddings(documents)

        self.embeddings = np.load(embeddings_path)
        if len(self.embeddings) != len(documents):
            raise ValueError("Embeddings count does not match documents")

        return self.embeddings

    def search(self, query: str, limit: int) -> dict[int, SemanticResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embeddings_query = self.model.encode([query], convert_to_numpy=True)[0]
        similarity_scores: list[float] = []

        for embedding in self.embeddings:
            similarity_scores.append(cosine_similarity(embeddings_query, embedding))

        if self.documents is None:
            raise ValueError("No documents loaded")
        score_to_doc = zip(similarity_scores, self.documents)

        sorted_scores = sorted(score_to_doc, key=lambda s: s[0], reverse=True)
        top_results = sorted_scores[:limit]
        return {
            doc["id"]: _format_search_result(
                title=doc["title"],
                document=doc["description"][:DOC_PREVIEW_LENGTH],
                score=score,
            )
            for score, doc in top_results
        }


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def _build_chunked_encodings(self, documents: Movies):
        all_chunks: list[str] = []
        all_chunk_metadata: list[dict[str, int]] = []

        for doc in documents:
            if doc["description"] == "":
                continue

            chunks = semantic_chunking(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for i in range(len(chunks)):
                all_chunk_metadata.append(
                    {
                        "movie_id": doc["id"],
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = all_chunk_metadata

        chunk_embeddings_path = CACHE_DIR.joinpath("chunk_embeddings.npy")
        np.save(chunk_embeddings_path, self.chunk_embeddings)

        chunk_metadata_path = CACHE_DIR.joinpath("chunk_metadata.json")
        with chunk_metadata_path.open("w") as f:
            json.dump(
                {"chunks": all_chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    @override
    def load_or_create_embeddings(self, documents: Movies):
        CACHE_DIR.mkdir(exist_ok=True)

        self.documents: Movies = documents
        self.document_map: dict[int, Movie] = {doc["id"]: doc for doc in documents}

        chunk_embeddings_path = CACHE_DIR.joinpath("chunk_embeddings.npy")
        if not chunk_embeddings_path.exists():
            return self._build_chunked_encodings(documents)

        chunk_metadata_path = CACHE_DIR.joinpath("chunk_metadata.json")
        if not chunk_metadata_path.exists():
            return self._build_chunked_encodings(documents)

        self.chunk_embeddings = np.load(chunk_embeddings_path)
        self.chunk_metadata = json.load(chunk_metadata_path.open())["chunks"]

        return self.chunk_embeddings

    def _compare_query_embedding_to_embedded_chunks(
        self, embedded_query: np.ndarray
    ) -> list[SimilarityScore]:
        if self.chunk_metadata is None or self.chunk_embeddings is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_embeddings` first."
            )

        similarity_scores: list[SimilarityScore] = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            cos_sim: float = cosine_similarity(embedded_query, chunk_embedding)

            d: SimilarityScore = {
                "chunk_idx": i,
                "movie_id": self.chunk_metadata[i]["movie_id"],
                "score": cos_sim,
            }
            similarity_scores.append(d)

        return similarity_scores

    def search_chunks(self, query: str, limit: int = 10):
        embedded_query: np.ndarray = super().generate_embedding(query)
        similarity_scores: list[SimilarityScore] = (
            self._compare_query_embedding_to_embedded_chunks(embedded_query)
        )

        movie_idx_to_score: defaultdict[int, float] = defaultdict(float)
        for item in similarity_scores:
            movie_id: int = item["movie_id"]

            movie_idx_to_score[movie_id] = max(
                movie_idx_to_score[movie_id], item["score"]
            )

        sorted_movies: list[tuple[int, float]] = sorted(
            movie_idx_to_score.items(), key=lambda x: x[1], reverse=True
        )
        top_movies: list[tuple[int, float]] = sorted_movies[:limit]

        results: dict[int, SemanticResult] = {}
        for movie_id, score in top_movies:
            document: str = self.document_map[movie_id]["description"]

            results[movie_id] = _format_search_result(
                self.document_map[movie_id]["title"],
                document[:DOC_PREVIEW_LENGTH] + "...",
                score,
            )

        return results
