"""Unit and regression tests for Stage 2 layout detection and segmentation."""
from pathlib import Path
import pytest
import numpy as np
import cv2

from doc_agent.contracts import Page, Region
from doc_agent.vision import layout


@pytest.fixture
def sample_synthetic_page(tmp_path: Path) -> Page:
    """Create a clean synthetic document page with heading, text, math fraction, and diagram."""
    height, width = 1200, 900
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 1. Large Top Heading
    cv2.putText(img, "Chapter 1: Real Numbers", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    # 2. Left-aligned Text Lines
    cv2.putText(img, "Let a and b be two distinct rational numbers.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "Then their arithmetic mean is given by:", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # 3. Indented Multi-tier Math Fraction (Numerator + Fraction Bar + Denominator)
    # Numerator
    cv2.putText(img, "a + b", (250, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    # Fraction bar
    cv2.line(img, (230, 355), (330, 355), 0, 2)
    # Denominator
    cv2.putText(img, "2", (275, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # 4. Geometric Figure / Diagram on the Right
    cv2.circle(img, (700, 300), 70, 0, 2)
    cv2.line(img, (630, 300), (770, 300), 0, 2)
    cv2.putText(img, "r", (705, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    # 5. Caption
    cv2.putText(img, "Fig 1.1: Circle with radius r", (600, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    img_path = tmp_path / "synthetic_page_0001.png"
    cv2.imwrite(str(img_path), img)

    return Page(id="test_doc__page_0001", image_path=str(img_path), doc_id="test_doc")


def test_layout_detect_contracts(sample_synthetic_page: Page):
    """Verify detect() returns list[Region] conforming to contracts.py."""
    cfg = {"layout": {}}
    regions = layout.detect([sample_synthetic_page], cfg)

    assert isinstance(regions, list)
    assert len(regions) > 0

    valid_kinds = {"heading", "text", "math", "figure", "table"}
    for r in regions:
        assert isinstance(r, Region)
        assert r.page_id == sample_synthetic_page.id
        assert len(r.bbox) == 4
        x, y, w, h = r.bbox
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert r.kind in valid_kinds


def test_layout_detect_kinds(sample_synthetic_page: Page):
    """Verify heading, text, and figure detection on synthetic page."""
    cfg = {"layout": {}}
    regions = layout.detect([sample_synthetic_page], cfg)

    kinds = [r.kind for r in regions]
    assert "heading" in kinds, "Failed to detect top heading"
    assert "text" in kinds, "Failed to detect prose text lines"
    assert "figure" in kinds, "Failed to detect geometric diagram"


def test_fraction_stack_merging():
    """Verify that vertically stacked numerator and denominator lines merge."""
    items = [
        {"x": 200, "y": 300, "w": 100, "h": 30, "fill": 0.12},
        {"x": 200, "y": 340, "w": 100, "h": 30, "fill": 0.12},
    ]
    merged = layout._merge_fraction_stacks(items, med_h=45.0)
    assert len(merged) == 1
    assert merged[0]["is_math"] is True
    assert merged[0]["h"] >= 65


def test_two_column_reading_order():
    """Verify 2-column exercise pages are sorted column-by-column."""
    # 5 lines in column 1 (left), 5 lines in column 2 (right)
    col1 = [{"x": 80, "y": 100 * i, "w": 250, "h": 35} for i in range(1, 6)]
    col2 = [{"x": 550, "y": 100 * i, "w": 250, "h": 35} for i in range(1, 6)]
    # Mix them up
    mixed = [col1[0], col2[0], col1[1], col2[1], col1[2], col2[2], col1[3], col2[3], col1[4], col2[4]]

    ordered = layout._sort_reading_order(mixed, width=900, height=1200)

    # First 5 items should all be column 1
    for item in ordered[:5]:
        assert item["x"] < 450
    # Next 5 items should all be column 2
    for item in ordered[5:]:
        assert item["x"] >= 450
