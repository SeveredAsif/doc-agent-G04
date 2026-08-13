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
        self.backend = self.cfg.get("backend", "trocr").lower()

        if self.backend == "tesseract":
            from pathlib import Path
            import shutil

            configured_path = self.cfg.get("tesseract_cmd")
            default_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            discovered_path = shutil.which("tesseract")
            self.tesseract_cmd = str(
                Path(configured_path)
                if configured_path
                else Path(discovered_path) if discovered_path else default_path
            )
            if not Path(self.tesseract_cmd).is_file():
                raise FileNotFoundError(
                    "Could not find Tesseract. Set ocr.tesseract_cmd to tesseract.exe."
                )
            self.tessdata_dir = self.cfg.get("tessdata_dir", "data/interim/tessdata")
            self.tesseract_lang = self.cfg.get("languages", "ben+eng")
            self.tesseract_psm = int(self.cfg.get("psm", 7))
            return

        import torch
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        model_name = self.cfg["model"]
        if self.backend != "trocr":
            raise ValueError("ocr.backend must be 'trocr' or 'tesseract'.")
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

        max_pixels = int(self.cfg.get("max_crop_pixels", 2_000_000))
        if crop.width * crop.height > max_pixels:
            scale = (max_pixels / (crop.width * crop.height)) ** 0.5
            new_width = max(1, int(round(crop.width * scale)))
            new_height = max(1, int(round(crop.height * scale)))
            crop = crop.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if self.backend == "tesseract":
            from io import BytesIO
            import subprocess

            image_bytes = BytesIO()
            crop.save(image_bytes, format="PNG", dpi=(300, 300))
            command = [
                self.tesseract_cmd,
                "stdin",
                "stdout",
                "--tessdata-dir",
                self.tessdata_dir,
                "-l",
                self.tesseract_lang,
                "--psm",
                str(self.tesseract_psm),
                "--oem",
                "1",
            ]
            completed = subprocess.run(
                command,
                input=image_bytes.getvalue(),
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Tesseract failed for {region.page_id}: {detail}")
            return completed.stdout.decode("utf-8", errors="replace").strip()

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
