from sentence_transformers import CrossEncoder
import logging

from .hybrid_search import RRFSearchResult


logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)


def rerank_cross_encoder(
    query: str, docs: list[tuple[int, RRFSearchResult]], limit: int
) -> list[tuple[int, RRFSearchResult]]:
    logger.info("%d docs to cross encode", len(docs))
    cross_encoder = get_cross_encoder()

    pairs: list[tuple[str, str]] = []
    for _, doc in docs:
        doc_str = f"Title: {doc['title']}, Description: {doc['description']}"
        pairs.append((query, doc_str))

    scores = cross_encoder.predict(pairs)
    for i, (_, doc) in enumerate(docs):
        doc["rerank"] = scores[i]

    logger.info("results successfully cross_encoded. Sorting...")
    return sorted(docs, key=lambda item: item[1]["rerank"], reverse=True)[:limit]
