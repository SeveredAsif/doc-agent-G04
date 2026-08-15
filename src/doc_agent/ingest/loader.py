"""Stage 1 — load scanned page-images"""
from __future__ import annotations
from ..contracts import *  # noqa

def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page]. IMPLEMENT."""
    from pathlib import Path

    raw_dir = Path("data/raw")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    max_pages = cfg.get("max_pages") if isinstance(cfg, dict) else None

    image_extensions = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    image_paths = sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        ),
        key=lambda path: path.as_posix(),
    )

    if max_pages is not None:
        max_pages = int(max_pages)
        if max_pages <= 0:
            raise ValueError("max_pages must be positive when set")
        image_paths = image_paths[:max_pages]

    pages: list[Page] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(raw_dir)
        parent_doc_id = relative_path.parent.as_posix().replace("/", "__")
        if parent_doc_id == ".":
            # Flat corpora (no subfolder per book) still encode the document in
            # the filename, e.g. "higher_math_page_0001.png" / "math_page_0001.png".
            # Strip the trailing "_page_<n>" so pages from different books don't
            # collapse into one doc_id (breaks document-level splits, C3).
            import re

            match = re.match(r"^(.*)_page_\d+$", image_path.stem)
            doc_id = match.group(1) if match else "default"
        else:
            doc_id = parent_doc_id
        page_id = f"{doc_id}__{image_path.stem}"
        pages.append(
            Page(
                id=page_id,
                image_path=image_path.as_posix(),
                doc_id=doc_id,
            )
        )

    return pages
