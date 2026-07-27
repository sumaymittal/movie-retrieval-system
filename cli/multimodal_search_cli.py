import argparse
import logging
import pathlib
from typing import cast

from lib.multimodal_search import MultimodalSearch, verify_image_embedding
from lib.search_utils import DOC_PREVIEW_LENGTH, load_movies

logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_img_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    _ = verify_img_parser.add_argument(
        "image", type=str, help="Path to image to be analysed"
    )

    img_search_parser = subparsers.add_parser(
        "image_search", help="Search with image rather than query"
    )
    _ = img_search_parser.add_argument(
        "image", type=str, help="Path to image to be analysed"
    )

    return parser


def cmd_image_search(image_path: pathlib.Path):
    movies = load_movies()
    mm_search = MultimodalSearch(movies)

    results = mm_search.search_with_image(image_path)
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = pathlib.Path(cast(str, args.image))
            logger.debug("image_path parsed: %s", image_path)
            verify_image_embedding(image_path)

        case "image_search":
            image_path = pathlib.Path(cast(str, args.image))
            logger.debug("image_path parsed: %s", image_path)

            results: list[dict[str, str]] = cmd_image_search(image_path)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']})")
                print(
                    f"    {res['description'][:DOC_PREVIEW_LENGTH].replace('\n', ' ')}...\n"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
