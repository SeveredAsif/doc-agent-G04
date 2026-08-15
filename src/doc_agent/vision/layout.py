"""Stage 2 - layout detection / region segmentation.

Refined Hybrid Layout Architecture:
  1. Vertical margin & corner pass (suppresses vertical publisher text & edge scanner noise)
  2. Enclosing callout box frame suppression (removes outer border frames of full-width definition/theorem boxes)
  3. Strict figure & table isolation (isolates compact 2D drawings like circles, triangles, plots, and tables)
  4. Label hull absorption (absorbs tiny annotation labels <= 45px within 35px of diagram)
  5. Figure-subtracted text foreground mask (diagrams cannot corrupt line extraction)
  6. Horizontal projection line slicing with tall line / fraction preservation (up to 200px)
  7. Multi-line paragraph valley splitting (only for truly dense blocks > 200px)
  8. Intra-line column gutter splitting (clean 2-column exercise separation)
  9. Figure-overlap suppression (prevents text boxes from piercing or overlapping figures)
 10. Heading prominence thresholding (heading_factor = 1.45)
 11. Dynamic headroom & diacritic padding (protects Bengali matras & tall radicals)
 12. Deterministic reading order sorting
"""
from __future__ import annotations

import json
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


def _suppress_callout_box_borders(thresh_clean: np.ndarray, width: int, height: int) -> np.ndarray:
    """Suppress outer border frames of wide enclosing callout boxes (w >= 0.65 * width) while preserving internal text."""
    kh_box = cv2.getStructuringElement(cv2.MORPH_RECT, (max(160, int(width * 0.40)), 1))
    kv_box = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(100, int(height * 0.05))))
    h_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kh_box)
    v_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kv_box)

    box_lines = cv2.bitwise_or(h_lines, v_lines)
    box_cnts, _ = cv2.findContours(box_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    callout_mask = np.zeros_like(thresh_clean)

    for bc in box_cnts:
        bx, by, bw, bh = cv2.boundingRect(bc)
        if bw >= width * 0.65 and bh >= 100:
            cv2.rectangle(callout_mask, (max(0, bx - 4), max(0, by - 4)), (min(width, bx + bw + 4), min(height, by + bh + 4)), 255, 12)

    return cv2.bitwise_and(thresh_clean, thresh_clean, mask=cv2.bitwise_not(callout_mask))


def _isolate_figures_and_tables(
    thresh_clean: np.ndarray,
    width: int,
    height: int,
    fig_pad_x: int = 35,
    fig_pad_y: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Strictly isolate 2D geometric diagrams and tabular grids (zero false positives on math formulas or callout boxes)."""
    fig_boxes: list[dict[str, Any]] = []
    fig_mask = np.zeros_like(thresh_clean)

    # 1. Detect structural data tables via grid lines
    kh_table = cv2.getStructuringElement(cv2.MORPH_RECT, (max(180, int(width * 0.28)), 1))
    kv_table = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(120, int(height * 0.08))))
    h_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kh_table)
    v_lines = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kv_table)
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
                fig_boxes.append({"kind": "figure", "bbox": (x0, y0, x1 - x0, y1 - y0)})
                cv2.rectangle(fig_mask, (x0, y0), (x1, y1), 255, -1)

    # 2. Strict Geometric Drawing / Plot Detection
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_fig_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        c_area = cv2.contourArea(cnt)
        is_true_diagram = (
            (w >= 130 and h >= 100 and c_area >= 10000 and w < width * 0.65)
            or (w >= 160 and h >= 120 and (w * h >= 35000) and c_area >= 8000 and w < width * 0.65)
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
    min_pitch: int = 35,
) -> list[tuple[int, int]]:
    """Split dense multi-paragraph blocks (> 180px) at the deepest valleys between consecutive text line peaks."""
    h = bottom - top
    if h <= 180:
        return [(top, bottom)]

    roi = ink[top:bottom, :]
    hpp = np.sum(roi > 0, axis=1).astype(np.float32)
    smoothed = np.convolve(hpp, np.ones(7) / 7, mode="same")
    mean_ink = np.mean(smoothed)

    # 1. Find line peaks (prominent local maxima >= 0.35 * mean_ink)
    peaks: list[int] = []
    for y in range(10, h - 10):
        w_start = max(0, y - 12)
        w_end = min(h, y + 13)
        if smoothed[y] == np.max(smoothed[w_start:w_end]) and smoothed[y] >= 0.35 * mean_ink:
            if not peaks or (y - peaks[-1] >= min_pitch):
                peaks.append(y)

    if len(peaks) <= 1:
        return [(top, bottom)]

    # 2. Find the deepest valley between each pair of consecutive peaks
    valleys = [0]
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        valley_y = p1 + int(np.argmin(smoothed[p1:p2]))
        valleys.append(valley_y)
    valleys.append(h)

    runs: list[tuple[int, int]] = []
    for i in range(len(valleys) - 1):
        s_top = top + valleys[i]
        s_bottom = top + valleys[i + 1]
        if s_bottom - s_top >= 18:
            runs.append((s_top, s_bottom))

    return runs if runs else [(top, bottom)]


def _extract_all_lines(
    ink: np.ndarray,
    page_w: int,
    page_h: int,
    row_ink_fraction: float = 0.002,
    line_gap: int = 18,
    min_line_height: int = 10,
    max_single_line_height: int = 200,
    pad_y: int = 6,
    pad_x: int = 8,
    min_line_width_frac: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """Extract unbroken text/math lines with fraction preservation and intra-line column separation."""
    row_ink = np.count_nonzero(ink, axis=1)
    active_rows = row_ink >= max(4, int(page_w * row_ink_fraction))

    raw_runs = _runs(active_rows, max_gap=line_gap)
    if not raw_runs:
        return []

    single_line_heights = [b - t for t, b in raw_runs if 18 <= (b - t) <= 55]
    if single_line_heights:
        med_h = float(np.median(single_line_heights))
    else:
        med_h = float(np.median([(b - t) / max(1, round((b - t) / 45.0)) for t, b in raw_runs]))

    split_runs: list[tuple[int, int]] = []
    for top, bottom in raw_runs:
        lh = bottom - top
        if lh < min_line_height:
            continue
        if lh > max_single_line_height:
            sub_runs = _split_dense_runs_by_projection(ink, top, bottom)
            split_runs.extend(sub_runs)
        else:
            split_runs.append((top, bottom))

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


def _absorb_and_prune_stray_boxes(
    boxes: list[tuple[int, int, int, int]],
    max_vertical_gap: int = 22,
    min_noise_area: int = 600,
) -> list[tuple[int, int, int, int]]:
    """Absorb small stray fragments into vertically adjacent parent text lines, and prune isolated noise."""
    if not boxes:
        return []

    # Separate primary lines from candidate strays
    primary: list[list[int]] = []
    strays: list[tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        if w >= 120 or h >= 35:
            primary.append([x, y, w, h])
        else:
            strays.append((x, y, w, h))

    unabsorbed: list[tuple[int, int, int, int]] = []

    for sx, sy, sw, sh in strays:
        absorbed = False
        s_mid_x = sx + sw * 0.5
        for p in primary:
            px, py, pw, ph = p
            # Check horizontal alignment
            if (px - 15) <= s_mid_x <= (px + pw + 15):
                # Stray directly above parent line
                if 0 <= (py - (sy + sh)) <= max_vertical_gap:
                    new_y0 = min(py, sy)
                    new_y1 = max(py + ph, sy + sh)
                    new_x0 = min(px, sx)
                    new_x1 = max(px + pw, sx + sw)
                    p[0], p[1], p[2], p[3] = new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0
                    absorbed = True
                    break
                # Stray directly below parent line
                elif 0 <= (sy - (py + ph)) <= max_vertical_gap:
                    new_y0 = min(py, sy)
                    new_y1 = max(py + ph, sy + sh)
                    new_x0 = min(px, sx)
                    new_x1 = max(px + pw, sx + sw)
                    p[0], p[1], p[2], p[3] = new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0
                    absorbed = True
                    break

        if not absorbed:
            unabsorbed.append((sx, sy, sw, sh))

    # Prune tiny scanner noise specks from unabsorbed strays
    kept_strays = []
    for ux, uy, uw, uh in unabsorbed:
        if uw < 45 and uh < 25 and (uw * uh < min_noise_area):
            continue
        kept_strays.append((ux, uy, uw, uh))

    return [tuple(p) for p in primary] + kept_strays


def _is_math_operator(
    crop: np.ndarray,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    area: int,
    th: int,
    stats: np.ndarray,
    i: int,
) -> bool:
    """Return True if CC strictly matches a mathematical operator or symbol."""
    if area < 10 or cw < 4 or ch < 4:
        return False

    aspect = cw / max(1, ch)
    c_center_y = cy + ch * 0.5

    # 1. Equality '=': Two separate parallel horizontal bars vertically aligned
    if 1.1 <= aspect <= 4.5 and ch <= th * 0.30:
        for j in range(1, len(stats)):
            if j == i:
                continue
            ox, oy, ow, oh, _ = stats[j]
            oaspect = ow / max(1, oh)
            if 1.1 <= oaspect <= 4.5 and oh <= th * 0.30:
                h_overlap = min(cx + cw, ox + ow) - max(cx, ox)
                if h_overlap >= 0.65 * min(cw, ow) and 2 <= abs(oy - cy) <= int(th * 0.40):
                    return True

    # 2. Plus sign '+': Symmetric cross shape
    if 0.75 <= aspect <= 1.30 and (th * 0.25 <= ch <= th * 0.75) and (th * 0.25 <= c_center_y <= th * 0.75):
        cc_roi = crop[cy : cy + ch, cx : cx + cw]
        mid_row = cc_roi[ch // 2, :]
        mid_col = cc_roi[:, cw // 2]
        if np.count_nonzero(mid_row) >= cw * 0.60 and np.count_nonzero(mid_col) >= ch * 0.60:
            c_tl = np.count_nonzero(cc_roi[: max(1, ch // 3), : max(1, cw // 3)])
            c_tr = np.count_nonzero(cc_roi[: max(1, ch // 3), cw - max(1, cw // 3) :])
            c_bl = np.count_nonzero(cc_roi[ch - max(1, ch // 3) :, : max(1, cw // 3)])
            c_br = np.count_nonzero(cc_roi[ch - max(1, ch // 3) :, cw - max(1, cw // 3) :])
            if (c_tl + c_tr + c_bl + c_br) <= area * 0.25:
                return True

    # 3. Horizontal minus in math context
    if aspect >= 2.4 and ch <= max(7, int(th * 0.22)) and (th * 0.30 <= c_center_y <= th * 0.70) and cw >= 12:
        return True

    # 4. Relational '<', '>', or arrow '\to'
    if 0.6 <= aspect <= 2.2 and (th * 0.25 <= ch <= th * 0.70):
        cc_roi = crop[cy : cy + ch, cx : cx + cw]
        left_col_ink = np.count_nonzero(cc_roi[:, 0])
        right_col_ink = np.count_nonzero(cc_roi[:, -1])
        mid_col_ink = np.count_nonzero(cc_roi[:, cw // 2])
        if (left_col_ink >= 2 and right_col_ink == 1 and mid_col_ink <= 2) or (
            right_col_ink >= 2 and left_col_ink == 1 and mid_col_ink <= 2
        ):
            return True

    return False


def _classify_math_lines(
    text_mask: np.ndarray,
    classified_lines: list[dict[str, Any]],
    page_w: int,
    page_h: int,
    median_line_height: float,
) -> None:
    """Classify entire line objects as kind='math' based on fill ratio, maatra continuity, fractions, and operators."""
    for line in classified_lines:
        if line["kind"] != "text":
            continue
        tx, ty, tw, th = line["bbox"]
        if tw < 40 or th < 12:
            continue

        crop = text_mask[ty : ty + th, tx : tx + tw]
        nonzero_count = np.count_nonzero(crop)
        if nonzero_count < 20:
            continue

        fill_ratio = nonzero_count / max(1, tw * th)

        # Check Maatra continuity in top headroom
        head_top = max(1, int(th * 0.15))
        head_bot = min(th, int(th * 0.38))
        headroom = crop[head_top:head_bot, :]
        head_col_active = np.any(headroom > 0, axis=0)
        m_runs = _runs(head_col_active, max_gap=2)
        continuous_maatra_runs = [(s, e) for s, e in m_runs if (e - s) >= 22]
        maatra_ink_cols = sum((e - s) for s, e in continuous_maatra_runs)
        maatra_ratio = maatra_ink_cols / max(1, tw)

        # Check for multi-story fraction division bar
        has_fraction = False
        if th >= 38:
            kh_frac = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(th * 0.40)), 1))
            frac_layer = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kh_frac)
            frac_cnts, _ = cv2.findContours(frac_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for fc in frac_cnts:
                fx, fy, fw, fh = cv2.boundingRect(fc)
                if fw >= max(18, int(th * 0.35)):
                    above_ink = np.count_nonzero(crop[max(0, fy - int(th * 0.40)) : fy, fx : fx + fw])
                    below_ink = np.count_nonzero(crop[fy + fh : min(th, fy + fh + int(th * 0.40)), fx : fx + fw])
                    if above_ink >= 12 and below_ink >= 12:
                        has_fraction = True
                        break

        # Check for explicit operators
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
        has_math_operators = False
        if num_labels > 1:
            for i in range(1, num_labels):
                cx, cy, cw, ch, area = stats[i]
                if _is_math_operator(crop, cx, cy, cw, ch, area, th, stats, i):
                    has_math_operators = True
                    break

        is_math = False
        # 1. Multi-story fractions with non-dense maatra
        if has_fraction and maatra_ratio < 0.35:
            is_math = True
        # 2. Low fill ratio (< 0.10) + confirmed math operators + low maatra (< 0.28)
        elif fill_ratio < 0.10 and has_math_operators and maatra_ratio < 0.28:
            is_math = True
        # 3. Very low maatra (< 0.12) + confirmed math operators
        elif maatra_ratio < 0.12 and has_math_operators:
            is_math = True
        # 4. Standalone / indented low-fill formula (fill < 0.08, maatra < 0.22, tw < page_w * 0.80)
        elif fill_ratio < 0.08 and maatra_ratio < 0.22 and tw < page_w * 0.80 and (has_math_operators or th >= 40):
            is_math = True

        if is_math:
            line["kind"] = "math"


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
    """Detect OCR-ready text lines, headings, figures, tables, and math expressions in reading order."""
    settings = cfg.get("layout", {})
    row_ink_fraction = float(settings.get("row_ink_fraction", 0.002))
    line_gap = int(settings.get("line_gap", 18))
    min_line_height = int(settings.get("min_line_height", 10))
    pad_y = int(settings.get("line_padding_y", 6))
    pad_x = int(settings.get("line_padding_x", 8))
    heading_factor = float(settings.get("heading_factor", 1.45))
    overlay_dir = settings.get("overlay_dir") or settings.get("debug_output_dir")
    save_overlays = bool(settings.get("save_overlays", True)) if overlay_dir else False
    metadata_path = settings.get("metadata_path")
    save_metadata = bool(settings.get("save_metadata", True)) if metadata_path else False
    detect_math = bool(settings.get("detect_math", True))

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
        thresh_no_margin = _vertical_margin_and_corner_pass(thresh, width, height)

        # 2. Suppress outer borders of wide enclosing callout frames
        thresh_clean = _suppress_callout_box_borders(thresh_no_margin, width, height)

        # 3. Strict Figure & Table isolation directly on cleaned image
        fig_table_boxes, fig_mask = _isolate_figures_and_tables(
            thresh_clean,
            width,
            height,
            fig_pad_x=35,
            fig_pad_y=20,
        )

        # 4. Figure-subtracted text ink
        text_only = cv2.bitwise_and(thresh_clean, thresh_clean, mask=cv2.bitwise_not(fig_mask))

        # 5. Extract unbroken lines with tall line / fraction preservation (line_gap = 18, max_single_line_height = 200)
        detected_line_boxes = _extract_all_lines(
            text_only,
            page_w=width,
            page_h=height,
            row_ink_fraction=row_ink_fraction,
            line_gap=line_gap,
            min_line_height=min_line_height,
            max_single_line_height=int(settings.get("max_single_line_height", 200)),
            pad_y=pad_y,
            pad_x=pad_x,
        )

        # 6. Filter text lines that overlap detected figures (prevents line piercing)
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

        # 7. Stray fragment absorption & noise pruning
        final_line_boxes = _absorb_and_prune_stray_boxes(clean_line_boxes)

        median_height = float(np.median([box[3] for box in final_line_boxes])) if final_line_boxes else 45.0

        # 8. Tag prominent headings (heading_factor = 1.45, top of page)
        classified_lines: list[dict[str, Any]] = []
        for index, bbox in enumerate(final_line_boxes):
            x, y, w, h = bbox
            if y < height * 0.12 and w >= width * 0.25 and h >= median_height * heading_factor:
                kind = "heading"
            else:
                kind = "text"

            classified_lines.append({"bbox": bbox, "kind": kind, "x": x, "y": y, "w": w, "h": h})

        # 9. Math Detection (Whole-Line math classification)
        if detect_math:
            _classify_math_lines(text_only, classified_lines, width, height, median_height)

        # 10. Combine all elements and sort reading order
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

        # 11. Compute Quantitative Page-Level Area Metrics
        total_page_area = int(width * height)
        fig_boxes = [e for e in ordered_elements if e["kind"] == "figure"]
        math_boxes = [e for e in ordered_elements if e["kind"] == "math"]
        text_boxes = [e for e in ordered_elements if e["kind"] in ("text", "heading")]
        heading_boxes = [e for e in ordered_elements if e["kind"] == "heading"]

        fig_area_px = int(sum(int(e["w"]) * int(e["h"]) for e in fig_boxes))
        math_area_px = int(sum(int(e["w"]) * int(e["h"]) for e in math_boxes))
        text_area_px = int(sum(int(e["w"]) * int(e["h"]) for e in text_boxes))

        page_metrics = {
            "num_headings": int(len(heading_boxes)),
            "num_text": int(len([e for e in text_boxes if e["kind"] == "text"])),
            "num_math": int(len(math_boxes)),
            "num_figures": int(len(fig_boxes)),
            "fig_area_px": fig_area_px,
            "math_area_px": math_area_px,
            "text_area_px": text_area_px,
            "fig_area_frac": float(round(fig_area_px / max(1, total_page_area), 4)),
            "math_area_frac": float(round(math_area_px / max(1, total_page_area), 4)),
            "text_area_frac": float(round(text_area_px / max(1, total_page_area), 4)),
            "total_ink_bbox_frac": float(
                round((fig_area_px + math_area_px + text_area_px) / max(1, total_page_area), 4)
            ),
        }

        # 12. Save Region Metadata JSONL
        if metadata_path and save_metadata:
            page_record = {
                "page_id": str(page.id),
                "doc_id": str(page.doc_id),
                "image_path": str(page.image_path),
                "width": int(width),
                "height": int(height),
                "total_page_area": total_page_area,
                "regions": [
                    {
                        "bbox": [int(v) for v in elem["bbox"]],
                        "kind": str(elem["kind"]),
                        "area": int(elem["bbox"][2] * elem["bbox"][3]),
                    }
                    for elem in ordered_elements
                ],
                "metrics": page_metrics,
            }
            meta_file = Path(metadata_path)
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_file, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(page_record, default=int, ensure_ascii=False) + "\n")

        # 13. Save Bounding-Box Overlay Image
        if overlay_dir and save_overlays:
            color_map = {
                "heading": (255, 191, 0),       # Deep Sky Blue
                "text": (50, 205, 50),          # Lime Green
                "figure": (0, 0, 230),          # Red
                "math": (180, 0, 255),          # Electric Purple
            }
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for index, elem in enumerate(ordered_elements):
                bx, by, bw, bh = elem["bbox"]
                bkind = elem["kind"]
                color = color_map.get(bkind, (0, 255, 255))
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
                tag_label = "M" if bkind == "math" else bkind[0].upper()
                tag_str = f"{index}:{tag_label}"
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
            page_filename = f"{page.id.rsplit('__', 1)[-1]}.png" if "__" in page.id else f"{Path(page.image_path).stem}.png"
            output_path = Path(overlay_dir) / page.doc_id / page_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), overlay):
                raise OSError(f"Could not write layout overlay: {output_path}")

    return regions
