"""Stage 2 - layout detection / region segmentation.

Redesigned multi-pass layout analysis pipeline:
  1. Border frame & vertical margin suppression (removes edge text & callout frames)
  2. Non-text isolation (extracts figures, plots, diagrams with +25px label hull expansion & table grids)
  3. Adaptive horizontal line smearing & fraction bar re-assembly (merges multi-tier math lines)
  4. Dynamic headroom padding (preserves Bengali diacritics, superscripts, root tails without clipping)
  5. 2-column exercise detection & column-aware reading order sorting
  6. Multi-label region tagging (heading, text, math, figure, table)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
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


def _suppress_vertical_margin_text(thresh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Mask out extreme vertical running text in outer margins (page edges)."""
    cleaned = thresh.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        is_outer_margin = (x < width * 0.05) or (x + w > width * 0.95)
        is_tall_vertical = (h > 40) and (h / max(1, w) > 3.5)
        if is_outer_margin and is_tall_vertical:
            cleaned[labels == i] = 0
    return cleaned


def _suppress_callout_borders(thresh: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress outer rectangular callout border frames while preserving internal strokes."""
    kh_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(160, int(width * 0.28)), 1))
    kv_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(100, int(height * 0.08))))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kh_long)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kv_long)
    box_borders = cv2.bitwise_or(h_lines, v_lines)
    dilated_borders = cv2.dilate(box_borders, np.ones((5, 5), np.uint8), iterations=1)
    thresh_clean = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(dilated_borders))
    return thresh_clean, h_lines, v_lines


def _isolate_figures_and_tables(
    thresh_clean: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    width: int,
    height: int,
    min_fig_area: int = 12000,
    fig_fill_thresh: float = 0.20,
    fig_pad: int = 25,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Isolate diagrams, coordinate plots, and tabular grids, expanding figure bounds to absorb labels."""
    fig_boxes: list[dict[str, Any]] = []
    fig_mask = np.zeros_like(thresh_clean)

    # 1. Detect structural data tables via intersecting grid lines
    table_grid = cv2.bitwise_and(h_lines, v_lines)
    dilated_grid = cv2.dilate(table_grid, np.ones((9, 9), np.uint8), iterations=2)
    table_cnts, _ = cv2.findContours(dilated_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in table_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h >= 16000) and (w > 100) and (h > 50):
            roi_grid = dilated_grid[y : y + h, x : x + w]
            cell_cnts, _ = cv2.findContours(roi_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(cell_cnts) >= 4:
                x0, y0 = max(0, x - 8), max(0, y - 8)
                x1, y1 = min(width, x + w + 8), min(height, y + h + 8)
                fig_boxes.append({"kind": "table", "bbox": (x0, y0, x1 - x0, y1 - y0)})
                cv2.rectangle(fig_mask, (x0, y0), (x1, y1), 255, -1)

    # 2. Detect geometric diagrams and plots (sparse stroke drawings grouped via small dilation)
    dilated_fig = cv2.dilate(thresh_clean, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated_fig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_fig_area and w > 75 and h > 75 and (w / max(1, h)) < 3.5:
            # Check if already covered by a table
            roi_mask = fig_mask[y : y + h, x : x + w]
            if np.count_nonzero(roi_mask) > 0.5 * area:
                continue

            roi = thresh_clean[y : y + h, x : x + w]
            fill = np.count_nonzero(roi) / max(1, area)
            if fill < fig_fill_thresh:
                # Expand hull by fig_pad to absorb vertex letters (A, B, C), angles, and axis numbers
                x0, y0 = max(0, x - fig_pad), max(0, y - fig_pad)
                x1, y1 = min(width, x + w + fig_pad), min(height, y + h + fig_pad)
                fig_boxes.append({"kind": "figure", "bbox": (x0, y0, x1 - x0, y1 - y0)})
                cv2.rectangle(fig_mask, (x0, y0), (x1, y1), 255, -1)

    return fig_boxes, fig_mask


def _merge_fraction_stacks(line_items: list[dict[str, Any]], med_h: float) -> list[dict[str, Any]]:
    """Merge vertically stacked numerator and denominator lines separated by a fraction bar."""
    if not line_items:
        return []

    sorted_items = sorted(line_items, key=lambda it: (it["y"], it["x"]))
    merged: list[dict[str, Any]] = []
    used = set()

    for i in range(len(sorted_items)):
        if i in used:
            continue
        cur = sorted_items[i]
        x1, y1, w1, h1 = cur["x"], cur["y"], cur["w"], cur["h"]
        is_math = cur.get("is_math", False)

        for j in range(i + 1, len(sorted_items)):
            if j in used:
                continue
            nxt = sorted_items[j]
            x2, y2, w2, h2 = nxt["x"], nxt["y"], nxt["w"], nxt["h"]

            # Check horizontal overlap and vertical proximity
            h_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            min_w = min(w1, w2)
            v_gap = y2 - (y1 + h1)

            if (h_overlap >= 0.50 * min_w) and (0 <= v_gap <= max(18, int(med_h * 0.45))):
                nx = min(x1, x2)
                ny = min(y1, y2)
                nw = max(x1 + w1, x2 + w2) - nx
                nh = max(y1 + h1, y2 + h2) - ny
                x1, y1, w1, h1 = nx, ny, nw, nh
                is_math = True
                used.add(j)

        used.add(i)
        merged.append(
            {
                "x": x1,
                "y": y1,
                "w": w1,
                "h": h1,
                "fill": cur["fill"],
                "is_math": is_math,
            }
        )

    return merged


def _sort_reading_order(
    line_items: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    """Sort components column-by-column for 2-column exercise layouts, or top-to-bottom for 1-column."""
    if not line_items:
        return []

    # Check for 2-column layout: valley in the vertical projection near page center
    mid_left = int(width * 0.40)
    mid_right = int(width * 0.60)
    center_occupancy = 0
    total_lines = len(line_items)

    for it in line_items:
        # Check if line crosses the center line
        if (it["x"] < width * 0.48) and (it["x"] + it["w"] > width * 0.52):
            center_occupancy += 1

    is_two_column = (total_lines >= 8) and (center_occupancy <= max(1, total_lines * 0.12))

    if is_two_column:
        col_split = width * 0.50
        col1 = [it for it in line_items if it["x"] + it["w"] * 0.5 < col_split]
        col2 = [it for it in line_items if it["x"] + it["w"] * 0.5 >= col_split]
        col1.sort(key=lambda it: it["y"])
        col2.sort(key=lambda it: it["y"])
        return col1 + col2

    # Standard 1-column reading order
    return sorted(line_items, key=lambda it: (it["y"], it["x"]))


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect OCR-ready text/math/heading lines and figure/table regions in reading order."""
    settings = cfg.get("layout", {})
    kh_ratio = float(settings.get("kh_ratio", 0.038))
    min_fig_area = int(settings.get("min_fig_area", 12000))
    fig_fill_thresh = float(settings.get("fig_fill_thresh", 0.20))
    fig_pad = int(settings.get("fig_pad", 25))
    heading_factor = float(settings.get("heading_factor", 1.45))
    math_indent_ratio = float(settings.get("math_indent_ratio", 0.05))
    math_fill_thresh = float(settings.get("math_fill_thresh", 0.14))
    pad_y = int(settings.get("line_padding_y", 10))
    pad_x = int(settings.get("line_padding_x", 12))
    debug_output_dir = settings.get("debug_output_dir")

    regions: list[Region] = []

    for page in pages:
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read page image: {page.image_path}")

        height, width = image.shape

        # Binarize with Otsu / Adaptive Gaussian
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        # 1. Suppress vertical edge text and callout border frames
        thresh_no_margin = _suppress_vertical_margin_text(thresh, width, height)
        thresh_clean, h_lines, v_lines = _suppress_callout_borders(thresh_no_margin, width, height)

        # 2. Isolate figures & tables and generate masking layer
        fig_table_boxes, fig_mask = _isolate_figures_and_tables(
            thresh_clean,
            h_lines,
            v_lines,
            width,
            height,
            min_fig_area=min_fig_area,
            fig_fill_thresh=fig_fill_thresh,
            fig_pad=fig_pad,
        )

        # 3. Extract text & math lines from unmasked foreground ink
        text_only = cv2.bitwise_and(thresh_clean, thresh_clean, mask=cv2.bitwise_not(fig_mask))
        kh = max(20, int(width * kh_ratio))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
        dilated_h = cv2.dilate(text_only, kernel_h, iterations=1)

        text_cnts, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_lines: list[dict[str, Any]] = []

        for cnt in text_cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 25 and h > 10:
                roi = thresh_clean[y : y + h, x : x + w]
                fill = np.count_nonzero(roi) / max(1, w * h)
                candidate_lines.append({"x": x, "y": y, "w": w, "h": h, "fill": fill})

        # Calculate page median line height & left margin
        heights = [item["h"] for item in candidate_lines] if candidate_lines else [45.0]
        med_h = float(np.median(heights))

        wide_lines = [it["x"] for it in candidate_lines if it["w"] > width * 0.35]
        left_margin = float(np.percentile(wide_lines, 15)) if wide_lines else width * 0.08

        # 4. Merge tall mathematical fractions & radicals
        merged_lines = _merge_fraction_stacks(candidate_lines, med_h)

        # 5. Tag lines into heading, text, and math
        classified_lines: list[dict[str, Any]] = []
        for it in merged_lines:
            x, y, w, h, fill = it["x"], it["y"], it["w"], it["h"], it["fill"]
            is_indented = (x - left_margin) >= (width * math_indent_ratio)
            is_sparse_stroke = fill < math_fill_thresh

            # Heading: large font ratio OR top header banner
            if (h >= med_h * heading_factor and w >= width * 0.18) or (
                y < height * 0.10 and w >= width * 0.22 and h >= med_h * 1.20
            ):
                kind = "heading"
            # Math: fraction merge flag OR indented display formula OR tall equation
            elif it.get("is_math", False) or (is_indented and w <= width * 0.70 and is_sparse_stroke) or (
                h >= med_h * 1.35 and is_sparse_stroke
            ):
                kind = "math"
            else:
                kind = "text"

            # Apply headroom and horizontal padding (clamped to page dimensions)
            px0 = max(0, x - pad_x)
            py0 = max(0, y - pad_y)
            px1 = min(width, x + w + pad_x)
            py1 = min(height, y + h + pad_y)

            classified_lines.append(
                {
                    "bbox": (px0, py0, px1 - px0, py1 - py0),
                    "kind": kind,
                    "x": px0,
                    "y": py0,
                    "w": px1 - px0,
                    "h": py1 - py0,
                }
            )

        # 6. Sort reading order across all elements (lines + tables + figures)
        all_elements = classified_lines + [
            {
                "bbox": fb["bbox"],
                "kind": fb["kind"],
                "x": fb["bbox"][0],
                "y": fb["bbox"][1],
                "w": fb["bbox"][2],
                "h": fb["bbox"][3],
            }
            for fb in fig_table_boxes
        ]

        ordered_elements = _sort_reading_order(all_elements, width, height)

        for elem in ordered_elements:
            regions.append(Region(page_id=page.id, bbox=elem["bbox"], kind=elem["kind"]))

        # 7. Debug Output Overlay Visualization
        if debug_output_dir:
            color_map = {
                "heading": (255, 191, 0),   # Deep Sky Blue
                "text": (50, 205, 50),      # Lime Green
                "math": (0, 140, 255),      # Orange
                "figure": (0, 0, 230),      # Red
                "table": (0, 215, 255),     # Gold
            }
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for index, elem in enumerate(ordered_elements):
                bx, by, bw, bh = elem["bbox"]
                bkind = elem["kind"]
                color = color_map.get(bkind, (0, 255, 255))
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
                tag_str = f"{index}:{bkind[0].upper()}"
                cv2.putText(
                    overlay,
                    tag_str,
                    (bx, max(18, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            output_path = Path(debug_output_dir) / page.doc_id / f"{page.id.rsplit('__', 1)[-1]}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), overlay):
                raise OSError(f"Could not write layout overlay: {output_path}")

    return regions
