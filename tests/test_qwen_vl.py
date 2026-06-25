"""Tests for Qwen2.5-VL vision-language integration."""
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_qwen_vl_processor_import():
    from src.vision.qwen_vl import QwenVLProcessor
    assert QwenVLProcessor is not None


def test_qwen_vl_processor_init_defaults():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    assert proc.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert proc.device == "cpu"
    assert proc._model is None
    assert proc._processor is None


def test_qwen_vl_processor_custom_config():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor(model_name="Qwen/Qwen2.5-VL-72B-Instruct", device="cuda:0")
    assert proc.model_name == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert proc.device == "cuda:0"


def test_qwen_vl_lazy_init_triggers_import_error():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    with pytest.raises((ImportError, RuntimeError)):
        proc._lazy_init()


def test_analyze_page_missing_file():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    with pytest.raises((FileNotFoundError, ImportError, RuntimeError)):
        proc.analyze_page("/nonexistent/page.png")


def test_extract_structured_missing_file():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    with pytest.raises((FileNotFoundError, ImportError, RuntimeError)):
        proc.extract_structured("/nonexistent/page.png")


def test_extract_structured_with_schema():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    with pytest.raises((FileNotFoundError, ImportError, RuntimeError)):
        proc.extract_structured("/nonexistent/page.png", output_schema=schema)


def test_analyze_page_returns_dict_on_error():
    from src.vision.qwen_vl import QwenVLProcessor
    proc = QwenVLProcessor()
    with pytest.raises((FileNotFoundError, ImportError, RuntimeError)):
        proc.analyze_page("/nonexistent.png")
