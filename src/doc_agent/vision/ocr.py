"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from ..contracts import *  # noqa

# TrOCR-style (VisionEncoderDecoder) checkpoints this reader accepts. The Bengali-aware
# entry is a Swin encoder + multilingual-BERT decoder finetuned for Bangla text; its
# tokenizer/image-processor live in a sibling repo, not bundled with the model weights,
# so it's loaded via cfg['ocr']['processor'] instead of TrOCRProcessor.from_pretrained.
_TROCR_STYLE_PREFIXES = (
    "microsoft/trocr",
    "nightsagittariuswolf/SWIN_TrOCR_Bangla_model",
)

class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.preprocess_cfg = cfg.get("preprocess", {})
        self.device_name = cfg.get("device", "cpu")

        import torch
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        model_name = self.cfg["model"]
        if not model_name.startswith(_TROCR_STYLE_PREFIXES):
            raise ValueError("The OCR reader requires a TrOCR-style (VisionEncoderDecoder) model.")

        self.torch = torch
        self.device = torch.device(
            self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        cache_dir = self.cfg.get("model_cache_dir", "data/interim/models")
        processor_name = self.cfg.get("processor", model_name)
        if processor_name == model_name:
            self.processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            image_processor = AutoImageProcessor.from_pretrained(processor_name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(processor_name, cache_dir=cache_dir)
            self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

        # Some community checkpoints (e.g. the Bangla Swin+mBERT model) shipped a
        # generation config with `null` fields that newer transformers rejects
        # outright (`early_stopping must be a boolean or 'never'`). Patch those
        # fields with sane defaults before construction rather than failing to load.
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        if getattr(config, "early_stopping", False) is None:
            config.early_stopping = False
        if getattr(config, "length_penalty", 1.0) is None:
            config.length_penalty = 1.0
        if getattr(config, "no_repeat_ngram_size", 0) is None:
            config.no_repeat_ngram_size = 0
        if getattr(config, "num_beams", 1) is None:
            config.num_beams = 1
        if getattr(config, "max_length", None) is None:
            config.max_length = int(self.cfg.get("max_new_tokens", 256))

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
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

        # A small white margin prevents glyphs at segmentation boundaries from
        # being cut in half before the processor resizes the line crop.
        from PIL import ImageOps

        crop = ImageOps.expand(crop, border=int(self.cfg.get("crop_padding", 8)), fill="white")

        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
        max_new_tokens = int(self.cfg.get("max_new_tokens", 256))
        with self.torch.inference_mode():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Transcribe line regions, then preserve page reading order in chunks."""
    reader = Reader(cfg)
    by_page: dict[str, list[str]] = {}
    chunks: list[Chunk] = []
    for region in regions:
        if region.kind not in {"text", "heading"}:
            continue
        text = reader.transcribe_region(region)
        if not text:
            continue
        by_page.setdefault(region.page_id, []).append(text)

    # Layout returns regions in reading order. Joining line outputs here gives
    # Stage 4 enough context to form useful semantic retrieval chunks; keeping
    # one Chunk per line would otherwise leave the index with isolated words.
    for page_id, lines in by_page.items():
        doc_id = page_id.rsplit("__", maxsplit=1)[0]
        chunks.append(
            Chunk(
                id=f"{page_id}__region_0000",
                doc_id=doc_id,
                text="\n".join(lines),
                page_ids=[page_id],
            )
        )
    return chunks
