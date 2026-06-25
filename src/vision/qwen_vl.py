"""Qwen2.5-VL vision-language document understanding.

⚠️ Requires Pro verification — ML model integration, eval/no_grad patterns.
"""

import json
import re
from pathlib import Path
from typing import Optional


class QwenVLProcessor:
    """Qwen2.5-VL model for document page analysis."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )
            self._model.eval()
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers qwen-vl-utils torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2.5-VL: {e}")

    def _prepare_inputs(self, image_path: Path, prompt: str):
        from PIL import Image
        import torch

        image = Image.open(str(image_path))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def analyze_page(self, image_path: str | Path) -> dict:
        """Analyze a document page: describe layout, identify tables/figures."""
        import torch
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._lazy_init()

        inputs = self._prepare_inputs(image_path, (
            "Analyze this document page. Return a JSON object with keys:\n"
            "- layout_type: one of 'text', 'table', 'figure', 'mixed'\n"
            "- has_table: boolean\n- has_figure: boolean\n"
            "- column_count: int\n- headers: list of headers\n"
            "- description: brief description"
        ))

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512)
        response = self._processor.decode(outputs[0], skip_special_tokens=True)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"layout_type": "unknown", "description": response}

    def extract_structured(self, image_path: str | Path, output_schema: Optional[dict] = None) -> dict:
        """Extract structured information following a schema."""
        import torch
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._lazy_init()

        schema_prompt = f"Return a JSON object matching this schema: {output_schema}" if output_schema else "Extract all visible structured data."
        inputs = self._prepare_inputs(image_path, schema_prompt)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1024)
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        return {"raw_response": response}
