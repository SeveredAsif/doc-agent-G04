"""Stage 1 — load scanned page-images"""
from __future__ import annotations
from ..contracts import *  # noqa

def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page]. IMPLEMENT."""
    from pathlib import Path

    del cfg
    raw_dir = Path("data/raw")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    image_extensions = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    image_paths = sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        ),
        key=lambda path: path.as_posix(),
    )

    pages: list[Page] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(raw_dir)
        doc_id = relative_path.parent.as_posix().replace("/", "__")
        if doc_id == ".":
            doc_id = "default"
        page_id = f"{doc_id}__{image_path.stem}"
        pages.append(
            Page(
                id=page_id,
                image_path=image_path.as_posix(),
                doc_id=doc_id,
            )
        )

    return pages
