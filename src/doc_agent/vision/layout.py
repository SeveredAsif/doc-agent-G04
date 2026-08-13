"""Stage 2 - layout detection / segmentation.

The OCR model used in this project is a line recogniser, so a page-sized text
region is not a valid input to it. This baseline finds text lines with a
horizontal projection profile and returns them in reading order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Page, Region


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Return true runs in a 1-D mask, merging small gaps within a line."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    for index, value in enumerate(mask):
        if not value:
            continue
        if start is None or index - last > max_gap + 1:
            if start is not None:
                runs.append((start, last + 1))
            start = index
        last = index
    if start is not None:
        runs.append((start, last + 1))
    return runs


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect OCR-ready text/heading lines in top-to-bottom reading order.

    A full content bounding box works for a page OCR engine, but not TrOCR:
    it causes the decoder to attend to an arbitrary part of a page. Line
    segmentation is deliberately classical, reproducible, and model-free.
    """
    import cv2

    settings = cfg.get("layout", {})
    row_ink_fraction = float(settings.get("row_ink_fraction", 0.002))
    min_line_height = int(settings.get("min_line_height", 12))
    max_line_height = int(settings.get("max_line_height", 140))
    line_gap = int(settings.get("line_gap", 18))
    padding = int(settings.get("line_padding", 12))
    min_line_width_fraction = float(settings.get("min_line_width_fraction", 0.03))
    debug_output_dir = settings.get("debug_output_dir")
    regions: list[Region] = []

    for page in pages:
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read page image: {page.image_path}")

        candidates = [image]
        if settings.get("retry_90deg_rotation", True):
            candidates.extend(
                [
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
                ]
            )

        detected_line_boxes: list[tuple[int, int, int, int]] | None = None
        for candidate in candidates:
            foreground = cv2.threshold(
                candidate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
            height, width = foreground.shape

            border = max(5, min(height, width) // 100)
            ink = foreground.copy()
            ink[:border, :] = 0
            ink[height - border :, :] = 0
            ink[:, :border] = 0
            ink[:, width - border :] = 0
            row_ink = np.count_nonzero(ink, axis=1)
            active_rows = row_ink >= max(4, int(width * row_ink_fraction))

            line_boxes: list[tuple[int, int, int, int]] = []
            for top, bottom in _runs(active_rows, line_gap):
                if bottom - top < min_line_height or bottom - top > max_line_height:
                    continue
                columns = np.flatnonzero(np.any(ink[top:bottom] > 0, axis=0))
                if columns.size == 0:
                    continue
                left, right = int(columns[0]), int(columns[-1]) + 1
                if right - left < max(20, int(width * min_line_width_fraction)):
                    continue
                crop_left, crop_top = max(0, left - padding), max(0, top - padding)
                crop_right, crop_bottom = min(width, right + padding), min(height, bottom + padding)
                line_boxes.append((crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top))

            if line_boxes:
                detected_line_boxes = line_boxes
                break

        if detected_line_boxes is None:
            detected_line_boxes = []

        median_height = float(np.median([box[3] for box in detected_line_boxes])) if detected_line_boxes else 0.0
        for index, bbox in enumerate(detected_line_boxes):
            kind = "heading" if index == 0 and bbox[3] > median_height * 1.35 else "text"
            regions.append(Region(page_id=page.id, bbox=bbox, kind=kind))

        if debug_output_dir:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for index, bbox in enumerate(detected_line_boxes):
                x, y, box_width, box_height = bbox
                kind = "heading" if index == 0 and box_height > median_height * 1.35 else "text"
                color = (255, 120, 0) if kind == "heading" else (0, 180, 0)
                cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), color, 3)
                cv2.putText(
                    overlay,
                    str(index),
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            output_path = Path(debug_output_dir) / page.doc_id / f"{page.id.rsplit('__', 1)[-1]}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), overlay):
                raise OSError(f"Could not write layout overlay: {output_path}")

    return regions
