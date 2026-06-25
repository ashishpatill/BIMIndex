"""PaddleOCR integration for OCR-based text extraction."""

from pathlib import Path
from typing import Optional


class PaddleOCREngine:
    """Wrapper around PaddleOCR for document text extraction."""

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs):
        self.lang = lang
        self.use_gpu = use_gpu
        self._extra_kwargs = kwargs
        self._ocr = None

    def _lazy_init(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. "
                "Run: pip install paddleocr"
            )
        kwargs = dict(self._extra_kwargs)
        kwargs.setdefault("lang", self.lang)
        kwargs.setdefault("use_gpu", self.use_gpu)
        kwargs.setdefault("show_log", False)
        kwargs.setdefault("use_angle_cls", True)
        self._ocr = PaddleOCR(**kwargs)

    @property
    def ocr(self):
        self._lazy_init()
        return self._ocr

    def extract_text(self, image_path: str | Path) -> str:
        """Extract plain text from a document image.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text lines joined by newlines.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._lazy_init()
        result = self._ocr.ocr(str(image_path), cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            bbox, (text, confidence) = line
            if confidence is not None and confidence >= 0.5:
                lines.append(text)
        return "\n".join(lines)

    def extract_text_with_confidence(self, image_path: str | Path, min_confidence: float = 0.5) -> list[dict]:
        """Extract text with individual word confidences.

        Args:
            image_path: Path to the image file.
            min_confidence: Minimum confidence threshold (0-1).

        Returns:
            List of dicts with 'text' and 'confidence' keys.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._lazy_init()
        result = self._ocr.ocr(str(image_path), cls=True)
        if not result or not result[0]:
            return []
        elements = []
        for line in result[0]:
            bbox, (text, confidence) = line
            conf = float(confidence) if confidence is not None else 0.0
            if conf >= min_confidence:
                elements.append({"text": text, "confidence": conf})
        return elements

    def extract_layout(self, image_path: str | Path) -> list[dict]:
        """Extract text with full layout information (bounding boxes, confidence).

        Args:
            image_path: Path to the image file.

        Returns:
            List of dicts with 'text', 'confidence', and 'bbox' keys.
            bbox is a flat list of 4 floats: [x1, y1, x2, y2, x3, y3, x4, y4].
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._lazy_init()
        result = self._ocr.ocr(str(image_path), cls=True)
        if not result or not result[0]:
            return []
        elements = []
        for line in result[0]:
            bbox, (text, confidence) = line
            elements.append({
                "text": text,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "bbox": [float(x) for point in bbox for x in point],
            })
        return elements
