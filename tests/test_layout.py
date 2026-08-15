"""Unit and regression tests for Stage 2 layout detection and segmentation."""
from pathlib import Path
import pytest
import numpy as np
import cv2

from doc_agent.contracts import Page, Region
from doc_agent.vision import layout


@pytest.fixture
def sample_synthetic_page(tmp_path: Path) -> Page:
    """Create a clean synthetic document page with heading, text, and diagram."""
    height, width = 1200, 900
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 1. Large Top Heading
    cv2.putText(img, "Chapter 1: Real Numbers", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)

    # 2. Left-aligned Text Lines
    cv2.putText(img, "Let a and b be two distinct rational numbers.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "Then their arithmetic mean is given by:", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # 3. Math Equation Line
    cv2.putText(img, "Average = (a + b) / 2", (250, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

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

    valid_kinds = {"heading", "text", "figure", "table", "math"}
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


def test_dense_paragraph_splitting(tmp_path: Path):
    """Verify that multi-line dense paragraph blocks (> 180px) without 0-ink gaps are split into lines."""
    height, width = 800, 800
    img = np.ones((height, width), dtype=np.uint8) * 255
    # 6 lines spaced tightly in a large block (> 180px)
    for i in range(6):
        cv2.putText(img, f"Line {i+1} in a dense continuous paragraph block.", (50, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    img_path = tmp_path / "dense_paragraph.png"
    cv2.imwrite(str(img_path), img)

    page = Page(id="dense__p1", image_path=str(img_path), doc_id="dense")
    regions = layout.detect([page], {"layout": {}})

    # Should detect at least 5 distinct line regions
    assert len(regions) >= 5
    for r in regions:
        assert r.kind == "text"


def test_two_column_reading_order():
    """Verify 2-column exercise pages are sorted column-by-column."""
    # 5 lines in column 1 (left), 5 lines in column 2 (right)
    col1 = [{"x": 80, "y": 100 * i, "w": 250, "h": 35, "kind": "text", "bbox": (80, 100 * i, 250, 35)} for i in range(1, 6)]
    col2 = [{"x": 550, "y": 100 * i, "w": 250, "h": 35, "kind": "text", "bbox": (550, 100 * i, 250, 35)} for i in range(1, 6)]
    mixed = [col1[0], col2[0], col1[1], col2[1], col1[2], col2[2], col1[3], col2[3], col1[4], col2[4]]

    ordered = layout._sort_reading_order(mixed, width=900, height=1200)

    # First 5 items should all be column 1
    for item in ordered[:5]:
        assert item["x"] < 450
    # Next 5 items should all be column 2
    for item in ordered[5:]:
        assert item["x"] >= 450


def test_tall_fraction_line_preservation(tmp_path: Path):
    """Verify tall 2-story fraction lines (height > 60px) are kept intact and not sliced."""
    height, width = 600, 800
    img = np.ones((height, width), dtype=np.uint8) * 255
    cv2.putText(img, "x^2 + y^2", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.line(img, (180, 205), (350, 205), 0, 3)
    cv2.putText(img, "2a", (245, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)

    img_path = tmp_path / "fraction_tall.png"
    cv2.imwrite(str(img_path), img)

    page = Page(id="frac__p1", image_path=str(img_path), doc_id="frac")
    regions = layout.detect([page], {"layout": {}})

    text_regions = [r for r in regions if r.kind in {"text", "heading"}]
    assert len(text_regions) == 1, f"Expected 1 text line region, got {len(text_regions)}"
    assert text_regions[0].bbox[3] >= 60


def test_maatra_fill_math_detection(tmp_path: Path):
    """Verify that pure Bengali prose has 0 math boxes and standalone formulas are detected as math."""
    height, width = 800, 900
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 1. Pure Bengali line with strong continuous top maatra (Zero math)
    cv2.rectangle(img, (80, 100), (500, 108), 0, -1)
    cv2.putText(img, "Bengali text with continuous headline bar", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2)

    # 2. Standalone math line: (a + b)^2 = a^2 + 2ab + b^2 (low maatra, open fill)
    cv2.putText(img, "(a + b)^2 = a^2 + 2ab + b^2", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2)

    img_path = tmp_path / "maatra_test.png"
    cv2.imwrite(str(img_path), img)

    page = Page(id="maatra__p1", image_path=str(img_path), doc_id="maatra")
    regions = layout.detect([page], {"layout": {"detect_math": True}})

    kinds = [r.kind for r in regions]
    assert "text" in kinds
    assert "math" in kinds
    math_regions = [r for r in regions if r.kind == "math"]
    assert len(math_regions) == 1
    assert math_regions[0].bbox[2] >= 70


def test_math_debug_overlay_generation(tmp_path: Path):
    """Verify debug overlay renders with math colors when debug_output_dir is set."""
    height, width = 600, 800
    img = np.ones((height, width), dtype=np.uint8) * 255
    cv2.putText(img, "x^2 + y^2 = r^2", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    img_path = tmp_path / "debug_math.png"
    cv2.imwrite(str(img_path), img)

    debug_dir = tmp_path / "debug_out"
    page = Page(id="debug_doc__page_0001", image_path=str(img_path), doc_id="debug_doc")
    cfg = {"layout": {"detect_math": True, "debug_output_dir": str(debug_dir)}}
    regions = layout.detect([page], cfg)

    overlay_file = debug_dir / "debug_doc" / "page_0001.png"
    assert overlay_file.is_file(), f"Overlay file was not created: {overlay_file}"


