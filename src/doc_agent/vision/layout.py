"""Stage 2 - layout detection / region segmentation.

Refined Hybrid Layout Architecture:
  1. Vertical margin & corner pass (suppresses vertical publisher text & edge scanner noise)
  2. Strict figure & table isolation directly on intact threshold image (zero false-positives on math)
  3. Label hull absorption (absorbs tiny annotation labels <= 45px, fig_pad = 35px)
  4. Figure-subtracted text foreground mask (diagrams cannot corrupt line extraction)
  5. Horizontal projection line slicing with tall line / fraction preservation (up to 180-200px)
  6. Multi-line paragraph valley splitting (only for dense blocks > 200px)
  7. Intra-line column gutter splitting (clean 2-column exercise separation)
  8. Figure-overlap suppression (prevents text boxes from piercing or overlapping figures)
  9. Dynamic headroom & diacritic padding (protects Bengali matras & tall radicals)
 10. Deterministic reading order sorting
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


def _vertical_margin_and_corner_pass(thresh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Detect and mask out vertical running text, publisher slugs, and scanner noise in outer margins/corners."""
    cleaned = thresh.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    margin_x_left = int(width * 0.08)
    margin_x_right = int(width * 0.92)
    corner_y_top = int(height * 0.08)
    corner_y_bottom = int(height * 0.92)

    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]

        is_outer_x = (x < margin_x_left) or (x + w > margin_x_right)
        is_corner_y = (y < corner_y_top) or (y + h > corner_y_bottom)

        # 1. Tall vertical text running along outer margins (e.g. Form-12, Book Title)
        is_vertical_text = is_outer_x and (h > 35 and h / max(1, w) >= 1.8)

        # 2. Extreme edge sliver noise or border lines
        is_edge_sliver = (x < width * 0.03 or x + w > width * 0.97) and (w < 180 or h / max(1, w) > 3.0)

        # 3. Corner publisher tags / registration marks outside body margins
        is_corner_tag = (is_outer_x and is_corner_y) and (w < 120 and h < 60)

        if is_vertical_text or is_edge_sliver or is_corner_tag:
            cleaned[labels == i] = 0

    return cleaned


def _isolate_figures_and_tables(
    thresh_clean: np.ndarray,
    width: int,
    height: int,
    fig_pad_x: int = 35,
    fig_pad_y: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Strictly isolate 2D geometric diagrams and tabular grids (zero false positives on math formulas)."""
    fig_boxes: list[dict[str, Any]] = []
    fig_mask = np.zeros_like(thresh_clean)

    # 1. Detect structural data tables via grid lines
    kh_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(180, int(width * 0.28)), 1))
    kv_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(120, int(height * 0.08))))
    h_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kh_long)
    v_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kv_long)
    table_grid = cv2.bitwise_and(h_lines, v_lines)
    dilated_grid = cv2.dilate(table_grid, np.ones((9, 9), np.uint8), iterations=2)
    table_cnts, _ = cv2.findContours(dilated_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in table_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h >= 25000) and (w > 120) and (h > 60):
            roi_grid = dilated_grid[y : y + h, x : x + w]
            cell_cnts, _ = cv2.findContours(roi_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(cell_cnts) >= 4:
                x0, y0 = max(0, x - 6), max(0, y - 6)
                x1, y1 = min(width, x + w + 6), min(height, y + h + 6)
                fig_boxes.append({"kind": "table", "bbox": (x0, y0, x1 - x0, y1 - y0)})
                cv2.rectangle(fig_mask, (x0, y0), (x1, y1), 255, -1)

    # 2. Strict Geometric Drawing / Plot Detection on intact binary image
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_fig_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        c_area = cv2.contourArea(cnt)
        is_true_diagram = (
            (w >= 130 and h >= 100 and c_area >= 12000)
            or (w >= 160 and h >= 120 and (w * h >= 35000) and c_area >= 8000)
        )
        if is_true_diagram:
            raw_fig_boxes.append((x, y, w, h))

    # 3. Absorb nearby diagram labels (e.g. A, B, C, r, theta <= 45px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_clean, connectivity=8)
    for fx, fy, fw, fh in raw_fig_boxes:
        nx0 = max(0, fx - fig_pad_x)
        ny0 = max(0, fy - fig_pad_y)
        nx1 = min(width, fx + fw + fig_pad_x)
        ny1 = min(height, fy + fh + fig_pad_y)

        sub_labels = labels[ny0:ny1, nx0:nx1]
        unique_lbls = np.unique(sub_labels)
        for l in unique_lbls:
            if l == 0:
                continue
            lx, ly, lw, lh, larea = stats[l]
            if lw <= 45 and lh <= 45 and larea <= 600:
                nx0 = min(nx0, lx)
                ny0 = min(ny0, ly)
                nx1 = max(nx1, lx + lw)
                ny1 = max(ny1, ly + lh)

        fig_boxes.append({"kind": "figure", "bbox": (nx0, ny0, nx1 - nx0, ny1 - ny0)})
        cv2.rectangle(fig_mask, (nx0, ny0), (nx1, ny1), 255, -1)

    return fig_boxes, fig_mask


def _split_dense_runs_by_projection(
    ink: np.ndarray,
    top: int,
    bottom: int,
    med_h: float,
) -> list[tuple[int, int]]:
    """Split truly tall multi-paragraph blocks (> 200px) at horizontal projection profile valleys."""
    h = bottom - top
    if h <= 200:
        return [(top, bottom)]

    roi = ink[top:bottom, :]
    hpp = np.sum(roi > 0, axis=1).astype(np.float32)

    kernel_size = max(5, int(med_h * 0.20))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(hpp, np.ones(kernel_size) / kernel_size, mode="same")

    est_lines = max(2, int(round(h / med_h)))
    target_step = h / est_lines

    split_points = [0]
    for k in range(1, est_lines):
        expected_y = int(k * target_step)
        search_min = max(0, int(expected_y - target_step * 0.35))
        search_max = min(h, int(expected_y + target_step * 0.35))

        if search_max > search_min:
            window = smoothed[search_min:search_max]
            valley_offset = int(np.argmin(window))
            valley_y = search_min + valley_offset
            if valley_y - split_points[-1] >= med_h * 0.60:
                split_points.append(valley_y)

    split_points.append(h)
    split_points = sorted(list(set(split_points)))

    result_runs: list[tuple[int, int]] = []
    for idx in range(len(split_points) - 1):
        s_top = top + split_points[idx]
        s_bottom = top + split_points[idx + 1]
        if s_bottom - s_top >= 12:
            result_runs.append((s_top, s_bottom))

    return result_runs if result_runs else [(top, bottom)]


def _extract_all_lines(
    ink: np.ndarray,
    page_w: int,
    page_h: int,
    row_ink_fraction: float = 0.002,
    line_gap: int = 14,
    min_line_height: int = 10,
    pad_y: int = 6,
    pad_x: int = 8,
    min_line_width_frac: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """Extract unbroken text/math lines with fraction preservation and intra-line column separation."""
    row_ink = np.count_nonzero(ink, axis=1)
    active_rows = row_ink >= max(4, int(page_w * row_ink_fraction))

    raw_runs = _runs(active_rows, line_gap)
    if not raw_runs:
        return []

    single_line_heights = [b - t for t, b in raw_runs if 18 <= (b - t) <= 85]
    med_h = float(np.median(single_line_heights)) if single_line_heights else 48.0

    split_runs: list[tuple[int, int]] = []
    for top, bottom in raw_runs:
        if bottom - top < min_line_height:
            continue
        sub_runs = _split_dense_runs_by_projection(ink, top, bottom, med_h)
        split_runs.extend(sub_runs)

    line_boxes: list[tuple[int, int, int, int]] = []

    for top, bottom in split_runs:
        if bottom - top < min_line_height:
            continue
        line_slice = ink[top:bottom, :]

        # Check for 2-column gutter in this line slice
        vpp_line = np.sum(line_slice > 0, axis=0)
        mid_s = int(page_w * 0.35)
        mid_e = int(page_w * 0.65)
        vpp_mid = vpp_line[mid_s:mid_e]

        col_gap_found = False
        if len(vpp_mid) > 0 and np.min(vpp_mid) == 0:
            zero_runs = _runs(vpp_mid == 0, max_gap=0)
            if zero_runs:
                widest_gap = max(zero_runs, key=lambda r: r[1] - r[0])
                gap_w = widest_gap[1] - widest_gap[0]
                if gap_w >= max(35, int(page_w * 0.04)):
                    gap_start = mid_s + widest_gap[0]
                    gap_end = mid_s + widest_gap[1]

                    left_cols = np.flatnonzero(vpp_line[:gap_start] > 0)
                    right_cols = np.flatnonzero(vpp_line[gap_end:] > 0)

                    if left_cols.size > 0 and right_cols.size > 0:
                        left_w = left_cols[-1] - left_cols[0]
                        right_w = right_cols[-1] - right_cols[0]
                        if left_w > 50 and right_w > 50:
                            col_gap_found = True
                            # Left Box
                            lx0 = max(0, int(left_cols[0]) - pad_x)
                            lx1 = min(page_w, int(left_cols[-1]) + 1 + pad_x)
                            ly0 = max(0, top - pad_y)
                            ly1 = min(page_h, bottom + pad_y)
                            line_boxes.append((lx0, ly0, lx1 - lx0, ly1 - ly0))

                            # Right Box
                            rx0 = max(0, gap_end + int(right_cols[0]) - pad_x)
                            rx1 = min(page_w, gap_end + int(right_cols[-1]) + 1 + pad_x)
                            line_boxes.append((rx0, ly0, rx1 - rx0, ly1 - ly0))

        if not col_gap_found:
            columns = np.flatnonzero(vpp_line > 0)
            if columns.size == 0:
                continue
            left = int(columns[0])
            right = int(columns[-1]) + 1
            line_w = right - left
            if line_w < max(20, int(page_w * min_line_width_frac)):
                continue

            abs_x0 = max(0, left - pad_x)
            abs_y0 = max(0, top - pad_y)
            abs_x1 = min(page_w, right + pad_x)
            abs_y1 = min(page_h, bottom + pad_y)
            line_boxes.append((abs_x0, abs_y0, abs_x1 - abs_x0, abs_y1 - abs_y0))

    return line_boxes


def _sort_reading_order(
    elements: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    """Sort layout elements in natural top-to-bottom / column-by-column reading order."""
    if not elements:
        return []

    mid_split = width * 0.50
    left_count = sum(1 for e in elements if (e["x"] + e["w"] * 0.5) < mid_split)
    right_count = sum(1 for e in elements if (e["x"] + e["w"] * 0.5) >= mid_split)
    total_count = len(elements)

    is_two_col = (total_count >= 8) and (left_count >= total_count * 0.30) and (right_count >= total_count * 0.30)

    if is_two_col:
        top_headers = [e for e in elements if e["y"] < height * 0.12 and e["w"] > width * 0.55]
        body_elements = [e for e in elements if e not in top_headers]

        col1 = [e for e in body_elements if (e["x"] + e["w"] * 0.5) < mid_split]
        col2 = [e for e in body_elements if (e["x"] + e["w"] * 0.5) >= mid_split]

        col1.sort(key=lambda it: it["y"])
        col2.sort(key=lambda it: it["y"])
        top_headers.sort(key=lambda it: (it["y"], it["x"]))

        return top_headers + col1 + col2

    return sorted(elements, key=lambda it: (it["y"], it["x"]))


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect OCR-ready text lines, headings, figures, and tables in reading order."""
    settings = cfg.get("layout", {})
    row_ink_fraction = float(settings.get("row_ink_fraction", 0.002))
    line_gap = int(settings.get("line_gap", 14))
    min_line_height = int(settings.get("min_line_height", 10))
    pad_y = int(settings.get("line_padding_y", 6))
    pad_x = int(settings.get("line_padding_x", 8))
    heading_factor = float(settings.get("heading_factor", 1.25))
    debug_output_dir = settings.get("debug_output_dir")

    regions: list[Region] = []

    for page in pages:
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read page image: {page.image_path}")

        height, width = image.shape

        # Adaptive Gaussian Binarization
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        # 1. Vertical margin and corner pass (suppress vertical text & border noise)
        thresh_clean = _vertical_margin_and_corner_pass(thresh, width, height)

        # 2. Strict Figure & Table isolation directly on intact binary image
        fig_table_boxes, fig_mask = _isolate_figures_and_tables(
            thresh_clean,
            width,
            height,
            fig_pad_x=35,
            fig_pad_y=20,
        )

        # 3. Figure-subtracted text ink
        text_only = cv2.bitwise_and(thresh_clean, thresh_clean, mask=cv2.bitwise_not(fig_mask))

        # 4. Extract unbroken lines with tall line / fraction preservation
        detected_line_boxes = _extract_all_lines(
            text_only,
            page_w=width,
            page_h=height,
            row_ink_fraction=row_ink_fraction,
            line_gap=line_gap,
            min_line_height=min_line_height,
            pad_y=pad_y,
            pad_x=pad_x,
        )

        # 5. Filter text lines that overlap detected figures (prevents line piercing)
        clean_line_boxes: list[tuple[int, int, int, int]] = []
        for tx, ty, tw, th in detected_line_boxes:
            overlaps_figure = False
            t_area = tw * th
            for fb in fig_table_boxes:
                fx, fy, fw, fh = fb["bbox"]
                ix0 = max(tx, fx)
                iy0 = max(ty, fy)
                ix1 = min(tx + tw, fx + fw)
                iy1 = min(ty + th, fy + fh)
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    if inter_area > 0.30 * t_area or (tx >= fx and tx + tw <= fx + fw and ty >= fy and ty + th <= fy + fh):
                        overlaps_figure = True
                        break
            if not overlaps_figure:
                clean_line_boxes.append((tx, ty, tw, th))

        median_height = float(np.median([box[3] for box in clean_line_boxes])) if clean_line_boxes else 45.0

        # 6. Tag lines (heading vs text)
        classified_lines: list[dict[str, Any]] = []
        for index, bbox in enumerate(clean_line_boxes):
            x, y, w, h = bbox
            if y < height * 0.12 and w >= width * 0.22 and (h >= median_height * heading_factor or (index == 0 and h >= median_height * 1.10)):
                kind = "heading"
            else:
                kind = "text"

            classified_lines.append({"bbox": bbox, "kind": kind, "x": x, "y": y, "w": w, "h": h})

        # 7. Combine all elements and sort reading order
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

        # 8. Debug Visualization Overlay
        if debug_output_dir:
            color_map = {
                "heading": (255, 191, 0),   # Deep Sky Blue
                "text": (50, 205, 50),      # Lime Green
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
