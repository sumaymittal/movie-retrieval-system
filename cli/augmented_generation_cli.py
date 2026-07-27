import argparse
import logging
from typing import cast, Callable

from lib.hybrid_search import HybridSearch, RRFSearchResults
from lib.search_utils import load_movies
from hybrid_search_cli import cmd_rrf_search
from lib.llm_utils import (
    PromptPair,
    get_answer_question_prompt,
    get_gemini_client,
    get_rag_citations_prompt,
    get_rag_nl_prompt,
    get_rag_summarize_prompt,
    query_gemini,
)

logger = logging.getLogger(__name__)

RAGPromptGenerator = Callable[[str, RRFSearchResults], PromptPair]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Get summary (search + generate summary)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")

    citations_parser = subparsers.add_parser(
        "citations", help="Get summary with citations (search + generate summary)"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")

    question_parser = subparsers.add_parser(
        "question", help="Get an answer to a question based on data in movie database"
    )
    question_parser.add_argument("query", type=str, help="Question to be answered")

    return parser


def cmd_rag(
    query: str, rag_prompter: RAGPromptGenerator
) -> tuple[RRFSearchResults, str]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    gemini_client = get_gemini_client()

    results = cmd_rrf_search(hybrid_search, query, limit=5)
    logger.info("rrf search results received")

    sys_prompt, contents = rag_prompter(query, results)
    rag_response = query_gemini(gemini_client, sys_prompt, contents)
    logger.info("rag response generated")

    return results, rag_response


def main():
    parser = get_parser()
    args = parser.parse_args()

    query = cast(str, args.query)

    match args.command:
        case "rag":
            results, rag_response = cmd_rag(query, get_rag_nl_prompt)
            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])
            print("\n RAG Response:\n", rag_response)

        case "summarize":
            results, summary = cmd_rag(query, get_rag_summarize_prompt)
            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])
            print("\n LLM Summary:\n", summary)

        case "citations":
            results, cited_summary = cmd_rag(query, get_rag_citations_prompt)
            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])
            print("\n LLM Summary:\n", cited_summary)

        case "question":
            results, answer = cmd_rag(query, get_answer_question_prompt)
            print("Search Results:")
            for _, res in results:
                print(" - ", res["title"])
            print("\n Answer:\n", answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
