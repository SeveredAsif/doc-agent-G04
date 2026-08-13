"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations
from ..contracts import *  # noqa


def _estimate_skew_angle(image: "np.ndarray") -> float:
    """Estimate the dominant page rotation in degrees, robust to 90° flips."""
    import cv2
    import numpy as np

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coordinates = cv2.findNonZero(foreground)
    if coordinates is None:
        return 0.0

    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle += 90.0
    elif angle > 45:
        angle -= 90.0

    # A very small skew is not worth rotating; 90° flips are the real problem.
    if abs(angle) < 1.0:
        return 0.0
    return float(angle)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing. IMPLEMENT."""
    from pathlib import Path

    import cv2

    settings = cfg.get("preprocess", {})
    output_dir = Path(settings.get("output_dir", "data/interim/preprocessed"))
    denoise_strength = settings.get("denoise_strength", 10)
    block_size = settings.get("adaptive_block_size", 31)
    threshold_offset = settings.get("adaptive_c", 15)
    max_pages = settings.get("max_pages")

    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("preprocess.adaptive_block_size must be an odd integer of at least 3")

    if max_pages is not None:
        max_pages = int(max_pages)
        if max_pages <= 0:
            raise ValueError("preprocess.max_pages must be positive when set")
        pages = pages[:max_pages]

    processed_pages: list[Page] = []
    for page in pages:
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read page image: {page.image_path}")

        angle = _estimate_skew_angle(image)
        if abs(angle) > 1.0:
            rotation = cv2.getRotationMatrix2D(
                (image.shape[1] / 2.0, image.shape[0] / 2.0),
                angle,
                1.0,
            )
            image = cv2.warpAffine(
                image,
                rotation,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        denoised = cv2.fastNlMeansDenoising(image, None, denoise_strength, 7, 21)
        processed = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            threshold_offset,
        )

        output_path = output_dir / page.doc_id / f"{Path(page.image_path).stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), processed):
            raise OSError(f"Could not write preprocessed page: {output_path}")
        processed_pages.append(
            page.model_copy(update={"image_path": output_path.as_posix()})
        )

    return processed_pages
