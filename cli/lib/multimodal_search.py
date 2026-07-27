import logging
import pathlib
from typing import Any

from lib.search_utils import DEFAULT_SEARCH_LIMIT, Movie, Movies
from lib.semantic_search import cosine_similarity
from numpy import ndarray
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_NAME = "clip-ViT-B-32"

logger = logging.getLogger(__name__)


def verify_image_embedding(image_path: pathlib.Path) -> None:
    mm_search = MultimodalSearch()

    embedding = mm_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


class MultimodalSearch:
    def __init__(self, docs: Movies = [], model_name=MODEL_NAME):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.docs: Movies = docs
        self.docmap: dict[int, Movie] = {int(doc["id"]): doc for doc in docs}

        doc_strings: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in docs
        ]
        self.text_embeddings: ndarray[Any] = self.model.encode(
            doc_strings, show_progress_bar=True
        )

    def embed_image(self, image_path: pathlib.Path):
        with Image.open(image_path) as im:
            # must pass to method as list, and we only want first (and only) element from response
            encoded_image = self.model.encode([im])[0]  # type: ignore[arg-type]
        logger.info("Image encoded")
        return encoded_image

    def search_with_image(
        self, image_path: pathlib.Path, limit=DEFAULT_SEARCH_LIMIT
    ) -> list[dict[str, str]]:
        embedded_image = self.embed_image(image_path)
        results: list[tuple[int, float]] = []

        for i, embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(embedding, embedded_image)
            id = self.docs[i]["id"]
            results.append((id, score))

        sorted_results = sorted(results, key=lambda res: res[1], reverse=True)[:limit]
        return [
            {
                "title": self.docmap[id]["title"],
                "description": self.docmap[id]["description"],
                "score": str(round(score, 3)),
            }
            for id, score in sorted_results
        ]
