"""Tests for PaddleOCR integration."""
from pathlib import Path
import sys

import pytest

# ---------------------------------------------------------------------------
# Module-level import helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_ocr_engine_import():
    """Verify the PaddleOCR engine module imports correctly."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    assert PaddleOCREngine is not None


def test_ocr_engine_init():
    """Verify engine initializes with defaults."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    assert engine.lang == "en"
    assert engine.use_gpu is False


def test_ocr_engine_custom_config():
    """Verify engine accepts custom configuration."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine(lang="ch", use_gpu=True)
    assert engine.lang == "ch"
    assert engine.use_gpu is True


def test_ocr_engine_extra_kwargs():
    """Verify engine passes extra kwargs through to PaddleOCR."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine(det_db_thresh=0.3, rec_batch_num=6)
    assert engine._extra_kwargs["det_db_thresh"] == 0.3
    assert engine._extra_kwargs["rec_batch_num"] == 6


def test_extract_text_handles_missing_file():
    """Verify graceful handling of missing image files."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    with pytest.raises(FileNotFoundError, match="Image not found"):
        engine.extract_text("/nonexistent/path.png")


def test_extract_text_with_confidence_handles_missing_file():
    """Verify extract_text_with_confidence handles missing files."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    with pytest.raises(FileNotFoundError, match="Image not found"):
        engine.extract_text_with_confidence("/nonexistent/path.png")


def test_extract_layout_handles_missing_file():
    """Verify extract_layout handles missing files."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    with pytest.raises(FileNotFoundError, match="Image not found"):
        engine.extract_layout("/nonexistent/path.png")


def test_lazy_init_not_called_on_construction():
    """Verify PaddleOCR is not imported at construction time."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    assert engine._ocr is None


def test_ocr_property_triggers_lazy_init():
    """Verify accessing the .ocr property triggers lazy initialization."""
    from src.ocr.paddle_ocr import PaddleOCREngine
    engine = PaddleOCREngine()
    with pytest.raises(ImportError, match="PaddleOCR not installed"):
        _ = engine.ocr
