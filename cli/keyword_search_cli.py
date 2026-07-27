#!/usr/bin/env python3

import argparse
import sys
import math
import time
from typing import cast

from lib.search_utils import (
    BM25_K1,
    BM25_B,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    process_string,
    print_search_results,
)

from lib.keyword_search import InvertedIndex


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser(
        "build", help="Build inverted index of movie titles and descriptions"
    )

    tf_parser = subparsers.add_parser(
        "tf", help="Get the Term Frequency of a term in a document"
    )
    _ = tf_parser.add_argument("doc_id", type=int, help="ID of doc to check")
    _ = tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser(
        "idf", help="Get the Inverse Document Frequency of a term in the dataset"
    )
    _ = idf_parser.add_argument("term", type=str, help="Search term")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get the TF-IDF of a term in a document"
    )
    _ = tf_idf_parser.add_argument("doc_id", type=int, help="ID of doc to check")
    _ = tf_idf_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get the BM25 IDF of a term in the dataset"
    )
    _ = bm25_idf_parser.add_argument("term", type=str, help="Search term")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    _ = bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    _ = bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    _ = bm25search_parser.add_argument("query", type=str, help="Search query")
    _ = bm25search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="How many results to display",
    )

    return parser


def cmd_build() -> None:
    print("Building index...")
    start = time.time()

    movies = load_movies()

    inv_idx = InvertedIndex()
    inv_idx.build(movies)
    inv_idx.save()

    end = time.time()

    print(f"Built index in {end - start}s")


def cmd_search(
    index: InvertedIndex, query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[tuple[int, str]]:

    processed_query = process_string()(query)

    search_results: list[tuple[int, str]] = []
    for q in processed_query:
        search_results.extend(index.get_document(q))

        if len(search_results) >= limit:
            search_results = search_results[:limit]
            break

    return search_results


def cmd_tf(index: InvertedIndex, doc_id: int, term: str) -> int:
    return index.get_tf(doc_id, term)


def cmd_idf(index: InvertedIndex, term: str) -> float:
    tokens = process_string()(term)
    if len(tokens) != 1:
        raise ValueError
    token = tokens.pop()

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.index.get(token, set()))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def cmd_tfidf(index: InvertedIndex, doc_id: int, term: str) -> float:
    tf = cmd_tf(index, doc_id, term)
    idf = cmd_idf(index, term)
    return tf * idf


def cmd_bm25_idf(index: InvertedIndex, term: str) -> float:
    return index.get_bm25_idf(term)


def cmd_bm25_tf(
    index: InvertedIndex, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    return index.get_bm25_tf(doc_id, term, k1, b)


def cmd_bm25_search(
    index: InvertedIndex, query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[tuple[int, str, float]]:
    return index.bm25_search(query, limit)


def _get_index() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files not found. Run 'build' command first.")
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)

    return index


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    match cast(str, args.command):
        case "build":
            cmd_build()

        case "search":
            index = _get_index()
            query = cast(str, args.query)
            search_results = cmd_search(index, query)
            print_search_results(query, search_results)

        case "tf":
            index = _get_index()
            doc_id, term = cast(int, args.doc_id), cast(str, args.term)
            tf = cmd_tf(index, doc_id, term)
            print(tf)

        case "idf":
            index = _get_index()
            term = cast(str, args.term)
            idf = cmd_idf(index, term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")

        case "tfidf":
            index = _get_index()
            doc_id, term = cast(int, args.doc_id), cast(str, args.term)
            tf_idf = cmd_tfidf(index, doc_id, term)
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            index = _get_index()
            term = cast(str, args.term)
            bm25_idf = cmd_bm25_idf(index, term)
            print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")

        case "bm25tf":
            index = _get_index()
            doc_id, term, k1, b = (
                cast(int, args.doc_id),
                cast(str, args.term),
                cast(float, args.k1),
                cast(float, args.b),
            )
            bm25tf = cmd_bm25_tf(index, doc_id, term, k1, b)
            print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

        case "bm25search":
            index = _get_index()
            query, limit = cast(str, args.query), cast(int, args.limit)

            results = cmd_bm25_search(index, query, limit)
            for i, (id, title, score) in enumerate(results, 1):
                print(f"{i}. ({id}) {title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
