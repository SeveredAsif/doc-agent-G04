"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from ..contracts import *  # noqa

class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.preprocess_cfg = cfg.get("preprocess", {})
        self.device_name = cfg.get("device", "cpu")

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if not self.cfg["model"].startswith("microsoft/trocr"):
            raise ValueError("The baseline OCR reader requires a Microsoft TrOCR model.")

        self.torch = torch
        self.device = torch.device(
            self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        cache_dir = self.cfg.get("model_cache_dir", "data/interim/models")
        self.processor = TrOCRProcessor.from_pretrained(self.cfg["model"], cache_dir=cache_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.cfg["model"], cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()

    def transcribe_region(self, region: Region) -> str:
        from pathlib import Path

        from PIL import Image

        doc_id, page_stem = region.page_id.rsplit("__", maxsplit=1)
        output_dir = Path(self.preprocess_cfg.get("output_dir", "data/interim/preprocessed"))
        image_path = output_dir / doc_id / f"{page_stem}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Could not find preprocessed page image: {image_path}")

        x, y, width, height = region.bbox
        with Image.open(image_path) as image:
            left = max(0, x)
            top = max(0, y)
            right = min(image.width, x + width)
            bottom = min(image.height, y + height)
            if right <= left or bottom <= top:
                raise ValueError(f"Invalid region bounding box for {region.page_id}: {region.bbox}")
            crop = image.convert("RGB").crop((left, top, right, bottom))

        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks. IMPLEMENT (calls Reader)."""
    reader = Reader(cfg)
    chunks: list[Chunk] = []
    for index, region in enumerate(regions):
        doc_id = region.page_id.rsplit("__", maxsplit=1)[0]
        chunks.append(
            Chunk(
                id=f"{region.page_id}__region_{index:04d}",
                doc_id=doc_id,
                text=reader.transcribe_region(region),
                page_ids=[region.page_id],
            )
        )
    return chunks
