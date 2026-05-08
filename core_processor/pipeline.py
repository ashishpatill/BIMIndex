from __future__ import annotations

import io

from PIL import Image
from google.genai import types

from core_processor.gemini_client import safe_generate_content
from core_processor.image_geometry import _crop_document, _resize_for_ocr
from core_processor.mlx_backend import glm_ocr_mlx
from core_processor.settings import GEMINI_MODEL
from retrieval_research.log import get_logger

_logger = get_logger("core_processor.pipeline")


def process_page(img: Image.Image, mode: str, system_prompt: str, user_query: str):
    try:
        _logger.info("Processing page — input size: %s", img.size)
        img = _crop_document(img)
        img = _resize_for_ocr(img)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        _logger.info("Image ready — size: %s, PNG: %.1f KB", img.size, len(img_bytes) / 1024)

        glm_output = ""
        final_output = ""

        if mode in ["Pure Local", "Hybrid"]:
            _logger.info("Running GLM-OCR via MLX...")
            glm_output = glm_ocr_mlx(img, system_prompt, user_query)
            _logger.info("GLM-OCR done — %d chars", len(glm_output))

        if mode in ["Pure Cloud", "Hybrid"]:
            if mode == "Hybrid":
                text = (
                    f"{system_prompt}\n\n{user_query}\n\n"
                    f"GLM-OCR raw output to refine:\n{glm_output}\n\n"
                    "Improve structure, fix errors, and reason step-by-step."
                )
            else:
                text = f"{system_prompt}\n\n{user_query}"

            _logger.info("Running Gemini (%s)...", GEMINI_MODEL)
            result = safe_generate_content(
                GEMINI_MODEL,
                [
                    types.Part.from_text(text=text),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
            )
            if result is not None:
                final_output = result
                _logger.info("Gemini done — %d chars", len(final_output))
            else:
                _logger.warning("Gemini returned no output, falling back to GLM-OCR output")
                final_output = glm_output
        else:
            final_output = glm_output

        return glm_output, final_output

    except Exception as exc:
        _logger.error("Page processing failed: %s", exc)
        return "", ""
