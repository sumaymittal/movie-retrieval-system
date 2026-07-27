#!/usr/bin/env python3

import argparse
from typing import cast

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)
from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    semantic_chunking,
    SemanticResult,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify loaded model")

    embded_text_parser = subparsers.add_parser(
        "embed_text", help="Embdeds inputted text to semantic vector"
    )
    _ = embded_text_parser.add_argument("text", type=str, help="Text to be embedded")

    _ = subparsers.add_parser(
        "verify_embeddings",
        help="Verifies the embeddings file and creates it if one does not exist",
    )

    embded_query_parser = subparsers.add_parser(
        "embedquery", help="Embdeds query to semantic vector"
    )
    _ = embded_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser(
        "search", help="Searches movie database for query"
    )
    _ = search_parser.add_argument("query", type=str, help="Query to be embedded")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of search results to print",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Input text to be chunked")
    _ = chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    _ = chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Number of words per chunk",
    )
    _ = chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap words per chunk",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Input text to be semantically chunked"
    )
    _ = semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    _ = semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_MAX_CHUNK_SIZE,
        help="Number of sentences per chunk",
    )
    _ = semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap words per chunk",
    )

    _ = subparsers.add_parser(
        "embed_chunks", help="Embeds movie database chunks to semantic vectors"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Searches chunked movie database for query"
    )
    _ = search_chunked_parser.add_argument(
        "query", type=str, help="Query to be searched"
    )
    _ = search_chunked_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of search results to print",
    )

    return parser


def cmd_verify():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def cmd_embed_text(text: str):
    search = SemanticSearch()
    return search.generate_embedding(text)


def cmd_verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cmd_search(query: str, limit: int) -> dict[int, SemanticResult]:
    search = SemanticSearch()

    documents = load_movies()
    _ = search.load_or_create_embeddings(documents)

    return search.search(query, limit)


def cmd_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    return semantic_chunking(text, chunk_size, overlap)


def cmd_semantic_chunk(
    text: str, chunk_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = 0
):
    return semantic_chunking(text, chunk_size, overlap)


def cmd_embed_chunks():
    search = ChunkedSemanticSearch()
    documents = load_movies()

    return search.load_or_create_embeddings(documents)


def cmd_search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = ChunkedSemanticSearch()
    documents = load_movies()

    _ = search.load_or_create_embeddings(documents)
    return search.search_chunks(query, limit)


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            cmd_verify()

        case "embed_text":
            text = cast(str, args.text)
            embedding = cmd_embed_text(text)

            print(f"Text: {text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")

        case "verify_embeddings":
            cmd_verify_embeddings()

        case "embedquery":
            query = cast(str, args.query)
            embedding = cmd_embed_text(query)

            print(f"Query: {query}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Shape: {embedding.shape}")

        case "search":
            query = cast(str, args.query)
            limit = cast(int, args.limit)
            search_results: dict[int, SemanticResult] = cmd_search(query, limit)

            for i, res in enumerate(search_results.values(), 1):
                print(f"{i}. {res['title']} (score: {res['score']})\n{res['document']}")

        case "chunk":
            text = cast(str, args.text)
            chunk_size = cast(int, args.chunk_size)
            overlap = cast(int, args.overlap)

            chunks = cmd_chunk(text, chunk_size, overlap)

            print(f"Chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            text = cast(str, args.text)
            chunk_size = cast(int, args.max_chunk_size)
            overlap = cast(int, args.overlap)

            chunks = cmd_semantic_chunk(text, chunk_size, overlap)

            print(f"Semantically chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            embeddings = cmd_embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            query = cast(str, args.query)
            limit = cast(int, args.limit)
            chunked_results: dict[int, SemanticResult] = cmd_search_chunked(
                query, limit
            )

            for i, res in enumerate(chunked_results.values(), 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
