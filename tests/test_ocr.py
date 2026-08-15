from pathlib import Path
import cv2
import numpy as np
import pytest

from doc_agent.contracts import Page, Region, Chunk
from doc_agent.vision import ocr


def test_ocr_transcribes_layout_regions(tmp_path):
    # Create synthetic test page image
    img_dir = tmp_path / "general_math"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "math_page_0001.png"

    img = np.full((600, 800), 255, dtype=np.uint8)
    # Line 1: Heading
    cv2.putText(img, "Chapter 1 Real Numbers", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    # Line 2: Text
    cv2.putText(img, "Introduction to algebra and sets.", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    # Line 3: Math
    cv2.putText(img, "x + y = 10", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.imwrite(str(img_path), img)

    page_id = "general_math__math_page_0001"
    regions = [
        Region(page_id=page_id, bbox=(40, 50, 500, 50), kind="heading"),
        Region(page_id=page_id, bbox=(40, 150, 600, 50), kind="text"),
        Region(page_id=page_id, bbox=(40, 250, 300, 50), kind="math"),
        Region(page_id=page_id, bbox=(40, 350, 200, 100), kind="figure"),  # should be skipped by OCR
    ]

    cfg = {
        "preprocess": {"output_dir": str(tmp_path)},
        "ocr": {
            "backend": "tesseract",
            "tessdata_dir": "data/interim/tessdata",
            "languages": "eng",
            "psm": 7,
            "crop_padding": 8,
        }
    }

    chunks = ocr.transcribe(regions, cfg)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_id == "general_math"
    assert chunk.page_ids == [page_id]
    assert len(chunk.text) > 0
    # Verify content extracted from the text lines
    assert "Chapter" in chunk.text or "Real" in chunk.text or "algebra" in chunk.text or "x" in chunk.text


def test_ocr_handles_invalid_or_out_of_bounds_bbox(tmp_path):
    img_dir = tmp_path / "general_math"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "math_page_0002.png"

    img = np.full((200, 200), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    page_id = "general_math__math_page_0002"
    regions = [
        Region(page_id=page_id, bbox=(0, 0, 0, 0), kind="text"),  # 0-width/height
        Region(page_id=page_id, bbox=(-10, -10, 5, 5), kind="text"),  # out-of-bounds
    ]

    cfg = {
        "preprocess": {"output_dir": str(tmp_path)},
        "ocr": {
            "backend": "tesseract",
            "tessdata_dir": "data/interim/tessdata",
            "languages": "eng",
            "psm": 7,
            "crop_padding": 4,
        }
    }

    # Should not crash with unhandled exception
    chunks = ocr.transcribe(regions, cfg)
    assert isinstance(chunks, list)
