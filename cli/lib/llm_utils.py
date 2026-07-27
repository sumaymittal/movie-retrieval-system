import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from typing import Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .hybrid_search import RRFSearchResult, RRFSearchResults

logger = logging.getLogger(__name__)

GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# Custom type for prompt content that can handle both strings and Part objects
PromptContent = Sequence[Union[str, types.Part]]
PromptPair = tuple[str, PromptContent]


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key is not None, "could not load gemini api key"

    return genai.Client(api_key=api_key)


def query_gemini(client: genai.Client, sys_prompt: str, contents: PromptContent) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
        ),
    )

    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

    if response.text:
        return response.text
    elif response.prompt_feedback is not None:
        logger.debug(response.prompt_feedback)
        raise RuntimeWarning("Text field was empty and response was blocked.")
    else:
        raise RuntimeWarning("No response received from gemini api")


def get_spelling_query(query: str) -> PromptPair:
    sys_prompt = """Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

If no errors, return the original query.
Corrected:"""

    return sys_prompt, [f'Query: "{query}"']


def get_rewritten_query(query: str) -> PromptPair:
    sys_prompt = """Rewrite this movie search query to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    return sys_prompt, [f'Original: "{query}"']


def get_expanded_query(query: str) -> PromptPair:
    sys_prompt = """Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
"""

    return sys_prompt, [f'Query: "{query}"']


def rerank_results_individual(
    client: genai.Client,
    query: str,
    docs: RRFSearchResults,
    limit: int,
) -> RRFSearchResults:
    logger.info("%d results to rerank", len(docs))

    sys_prompt = """Rate how well this movie matches the search query.

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

    for _, doc in docs:
        contents = [
            f'Query: "{query}"',
            f"Movie: {doc.get('title', '')} - {doc.get('document', '')}",
        ]

        new_score = int(query_gemini(client, sys_prompt, contents))
        doc["rerank"] = new_score
        time.sleep(12)
        logger.info(
            "%s got new score %d. Waiting 12 seconds before next request",
            doc["title"],
            new_score,
        )

    return sorted(docs, key=lambda item: item[1]["rrf"], reverse=True)[:limit]


def rerank_results_batch(
    client: genai.Client,
    query: str,
    docs: RRFSearchResults,
    limit: int,
) -> RRFSearchResults:
    logger.info("%d results to rerank. Making api call.", len(docs))

    docs_mapped: dict[int, RRFSearchResult] = {id: doc for id, doc in docs}
    doc_list_str = "\n\n".join(
        f"ID: {id}, Title: {doc['title']}, Description: {doc['description']}"
        for id, doc in docs
    )

    sys_prompt = """Rank these movies by relevance to the search query.

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    contents = [f'Query: "{query}"', f"Movies:\n{doc_list_str}"]

    reranked_list_str = query_gemini(client, sys_prompt, contents)
    reranked_list = json.loads(reranked_list_str)
    logger.info("reranked list received and loaded to json")

    reranked_results: list[tuple[int, RRFSearchResult]] = []
    for rank, res_id in enumerate(reranked_list, 1):
        doc = docs_mapped[res_id]
        doc["rerank"] = rank
        reranked_results.append((res_id, doc))

    return reranked_results[:limit]


def get_evaluate_prompt(
    query: str, results: list[tuple[int, RRFSearchResult]]
) -> PromptPair:
    formatted_results = "\n".join(
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    )

    sys_prompt = """Rate how relevant each result is to this query on a 0-3 scale:

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    return sys_prompt, [f'Query: "{query}"', f"Results:\n{formatted_results}"]


def get_evaluation(
    client: genai.Client, query: str, results: list[tuple[int, RRFSearchResult]]
):
    logger.info("%d results to evaluate. Making api call.", len(results))

    try:
        sys_prompt, contents = get_evaluate_prompt(query, results)
        evaluation_results_str = query_gemini(client, sys_prompt, contents)
        logger.debug("Evaluation results as string: %s", evaluation_results_str)
    except Exception as e:
        print(e)
        sys.exit()

    evaluation_results = json.loads(evaluation_results_str)
    logger.info("evaluations received and loaded to json")

    for i, ((_, doc), score) in enumerate(zip(results, evaluation_results), 1):
        print(f"{i}. {doc['title']}: {score}/3")


def get_rag_nl_prompt(query: str, results: RRFSearchResults) -> PromptPair:
    formatted_results = "\n".join(
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    )

    sys_prompt = """Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Provide a comprehensive answer that addresses the query:"""

    return sys_prompt, [f"Query: {query}", f"Documents:\n{formatted_results}"]


def get_rag_summarize_prompt(query: str, results: RRFSearchResults) -> PromptPair:
    formatted_results = "\n".join(
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    )

    sys_prompt = """Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    return sys_prompt, [f"Query: {query}", f"Search Results:\n{formatted_results}"]


def get_rag_citations_prompt(query: str, results: RRFSearchResults) -> PromptPair:
    formatted_results = "\n".join(
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    )

    sys_prompt = """Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    return sys_prompt, [f"Query: {query}", f"Documents:\n{formatted_results}"]


def get_answer_question_prompt(query: str, results: RRFSearchResults) -> PromptPair:
    formatted_results = "\n".join(
        f"Title: {doc['title']}, Description: {doc['description']}"
        for _, doc in results
    )

    sys_prompt = """Answer the following question based on the provided documents.

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""

    return sys_prompt, [f"Question: {query}", f"Documents:\n{formatted_results}"]


def get_image_search_prompt(query: str, image: bytes, mime: str) -> PromptPair:
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    parts: PromptContent = [
        system_prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        query.strip(),
    ]

    return system_prompt, parts
