"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions. IMPLEMENT."""
    import cv2

    del cfg
    regions: list[Region] = []
    for page in pages:
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read page image: {page.image_path}")

        foreground = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        coordinates = cv2.findNonZero(foreground)
        if coordinates is None:
            height, width = image.shape
            bbox = (0, 0, width, height)
        else:
            x, y, width, height = cv2.boundingRect(coordinates)
            bbox = (x, y, width, height)

        regions.append(Region(page_id=page.id, bbox=bbox, kind="text"))

    return regions
