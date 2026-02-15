"""Janulon CLI: scrape sources and (optionally) run the aesthetic pipeline."""

import argparse
from pathlib import Path

from loguru import logger
from retina import fourchan, tumblr


def _run_4chan(
    board: str,
    output_folder: Path,
    index_pages: int,
    data_dir: Path | None,
) -> None:
    logger.info(
        "Starting 4chan scrape: board=/{}/, output={}, index_pages={}",
        board,
        output_folder.resolve(),
        index_pages,
    )
    urls = fourchan.iter_image_urls(
        board=board,
        index_pages=index_pages,
    )
    if not urls:
        logger.warning("No image URLs found.")
        return
    skip_dirs = []
    if data_dir is not None:
        skip_dirs = [data_dir / "corpus", data_dir / "void"]
    paths = fourchan.download_images(
        urls, output_folder, board, skip_dirs=skip_dirs or None
    )
    logger.success(
        "Done. Downloaded {} images to {}", len(paths), output_folder.resolve()
    )


def _run_tumblr(
    blog: str,
    output_folder: Path,
    num_posts: int,
    data_dir: Path | None,
) -> None:
    logger.info(
        "Starting Tumblr scrape: blog={}, output={}, num_posts={}",
        blog,
        output_folder.resolve(),
        num_posts,
    )
    urls = tumblr.iter_image_urls(blog=blog, num_posts=num_posts)
    if not urls:
        logger.warning("No image URLs found.")
        return
    skip_dirs = []
    if data_dir is not None:
        skip_dirs = [data_dir / "corpus", data_dir / "void"]
    paths = tumblr.download_images(
        urls, output_folder, blog, skip_dirs=skip_dirs or None
    )
    logger.success(
        "Done. Downloaded {} images to {}", len(paths), output_folder.resolve()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Janulon: scrape image sources and filter by taste.",
    )
    parser.add_argument(
        "--source",
        choices=["4chan", "reddit", "tumblr"],
        required=True,
        help="Image source to scrape.",
    )
    parser.add_argument(
        "--board",
        default="wg",
        help="Board to scrape (4chan only). Default: wg (wallpaper general).",
    )
    parser.add_argument(
        "--blog",
        help="Tumblr blog to scrape (tumblr only). Example: staff or blogname.tumblr.com",
    )
    parser.add_argument(
        "--subreddit",
        help="Subreddit to scrape (reddit only).",
    )
    parser.add_argument(
        "--num_posts",
        type=int,
        default=50,
        metavar="N",
        help="Max number of posts to fetch (tumblr only). Default: 50",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("data/wg"),
        help="Folder to save images. Default: data/wg",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Check corpus/ and void/ under this dir; skip download if image already there. Default: data",
    )
    parser.add_argument(
        "--index_pages",
        type=int,
        default=2,
        metavar="N",
        help="Number of index pages to scrape (4chan only). Default: 2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for acceptance (future use). Default: 0.85",
    )
    args = parser.parse_args()

    if args.source == "4chan":
        _run_4chan(
            board=args.board,
            output_folder=args.output_folder,
            index_pages=args.index_pages,
            data_dir=args.data_dir,
        )
    elif args.source == "tumblr":
        if not args.blog:
            logger.error(
                "Tumblr source requires --blog (e.g. staff or blogname.tumblr.com)."
            )
            raise SystemExit(1)
        _run_tumblr(
            blog=args.blog.strip(),
            output_folder=args.output_folder,
            num_posts=args.num_posts,
            data_dir=args.data_dir,
        )
    elif args.source == "reddit":
        logger.error("Reddit source is not implemented (API key required).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
