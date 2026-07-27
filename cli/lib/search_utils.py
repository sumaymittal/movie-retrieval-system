import json
from pathlib import Path
import re
import string
from typing import Callable, TypedDict, cast

from nltk.stem import PorterStemmer
import numpy as np


class SimilarityScore(TypedDict):
    chunk_idx: int
    movie_id: int
    score: float


MOVIES_FILE_PATH = "./data/movies.json"
STOPWORDS_FILE_PATH = "./data/stopwords.txt"
CACHE_DIR = Path("./cache")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

DEFAULT_SEARCH_LIMIT = 5
DOC_PREVIEW_LENGTH = 100
SCORE_PRECISION = 4
DEFAULT_K = 60

DEFAULT_CHUNK_SIZE = 200
DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 2

BM25_K1 = 1.5
BM25_B = 0.75


class Movie(TypedDict):
    id: int
    title: str
    description: str


Movies = list[Movie]


class MovieData(TypedDict):
    movies: Movies


def load_movies() -> Movies:
    with open(MOVIES_FILE_PATH, "r") as f:
        movies_data = cast(MovieData, json.load(f))
    return movies_data["movies"]


def process_string() -> Callable[[str], list[str]]:
    def _load_stopwords() -> set[str]:
        with open(STOPWORDS_FILE_PATH, "r") as f:
            stopwords = f.read().splitlines()
        return set(stopwords)

    stopwords = _load_stopwords()
    stemmer = PorterStemmer()

    def wrapper(s: str) -> list[str]:
        punc_removed = s.lower().translate(_PUNCT_TABLE)
        tokens = punc_removed.split()

        filtered = [t for t in tokens if t and t not in stopwords]
        return [stemmer.stem(t) for t in filtered]

    return wrapper


def print_search_results(query: str, search_results: list[tuple[int, str]]) -> None:
    print(f"Searching for: {query}")
    for i, res in enumerate(search_results, 1):
        print(f"{i}. {res[1]}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return cast(float, dot_product / (norm1 * norm2))


def semantic_chunking(
    text: str, chunk_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = 0
) -> list[str]:
    """Split text into overlapping chunks of sentences.

    Args:
        text: Input text to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks (must be less than chunk_size)

    Returns:
        List of text chunks
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size")

    stripped = text.strip()
    if len(stripped) == 0:
        return []

    sentence_enders = (".", "!", "?")

    sentences: list[str] = re.split(r"(?<=[.!?])\s+", stripped)
    if len(sentences) == 1 and not sentences[0].endswith(sentence_enders):
        return [sentences[0]]

    # Filter out empty sentences
    filtered_sentences = [
        s.strip() for s in sentences if s.strip() not in sentence_enders
    ]

    chunks: list[str] = []
    i = 0
    while i < len(filtered_sentences):
        start = max(0, i - overlap)
        end = start + chunk_size

        chunk = " ".join(filtered_sentences[start:end])
        if chunk:
            chunks.append(chunk)
        i = end

    return chunks
