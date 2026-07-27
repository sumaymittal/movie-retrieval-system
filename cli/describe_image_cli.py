import argparse
import logging
import mimetypes
import pathlib as path
from typing import cast

from lib.llm_utils import get_gemini_client, get_image_search_prompt, query_gemini

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", type=str, help="Path to image to be analysed")
    parser.add_argument("--query", type=str, help="Query to be answered")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    query = cast(str, args.query)
    image_path = path.Path(cast(str, args.image))

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image = f.read()

    gemini_client = get_gemini_client()
    sys_prompt, content = get_image_search_prompt(query, image, mime)

    response = query_gemini(gemini_client, sys_prompt, content)

    print(f"Rewritten query: {response.strip()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
