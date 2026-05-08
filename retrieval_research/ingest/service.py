from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image, UnidentifiedImageError

from retrieval_research.config import get_settings
from retrieval_research.log import get_logger
from retrieval_research.profiling import build_document_profile
from retrieval_research.schema import Document, Page
from retrieval_research.storage import ArtifactStore

_logger = get_logger("ingest")


def stable_document_id(path: Path, content: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(str(path).encode("utf-8"))
    digest.update(content[:1024 * 1024])
    return digest.hexdigest()[:16]


def _document_from_text(path: Path, text: str, document_id: str) -> Document:
    return Document(
        id=document_id,
        source_path=str(path),
        title=path.stem,
        pages=[Page(id=f"{document_id}:page:1", number=1, text=text)],
        metadata={"source_type": path.suffix.lower().lstrip(".") or "text"},
    )


def _document_from_history(path: Path, document_id: str) -> Optional[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _logger.warning("Corrupt history JSON %s: %s", path.name, exc)
        return None
    pages: List[Page] = []
    page_idx = 1
    for file_data in payload.get("results", []):
        for page in file_data.get("pages", []):
            pages.append(
                Page(
                    id=f"{document_id}:page:{page_idx}",
                    number=page_idx,
                    text=page.get("final") or page.get("glm") or "",
                    metadata={
                        "source_filename": file_data.get("filename"),
                        "source_page": page.get("page"),
                        "glm_text": page.get("glm", ""),
                    },
                )
            )
            page_idx += 1
    return Document(
        id=document_id,
        source_path=str(path),
        title=path.stem,
        pages=pages,
        metadata={"source_type": "history_json", "original_timestamp": payload.get("timestamp")},
    )


def _save_image_page(store: ArtifactStore, document_id: str, image: Image.Image, page_number: int) -> str:
    page_path = store.page_dir(document_id) / f"page_{page_number:04d}.png"
    try:
        image.save(page_path, format="PNG")
    except Exception as exc:
        _logger.warning("Failed to save page image %s: %s", page_path, exc)
    return str(page_path)


def _ocr_image(image: Image.Image, mode: str, system_prompt: str, user_query: str) -> str:
    try:
        from core_processor import process_page

        _glm, final = process_page(image, mode, system_prompt, user_query)
        return final
    except Exception as exc:
        _logger.error("OCR processing failed: %s", exc)
        return ""


def _document_from_image(
    path: Path,
    content: bytes,
    document_id: str,
    store: ArtifactStore,
    run_ocr: bool,
    mode: str,
    system_prompt: str,
    user_query: str,
) -> Optional[Document]:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        _logger.error("Corrupt image %s: %s", path.name, exc)
        return None
    image_path = _save_image_page(store, document_id, image, 1)
    text = _ocr_image(image, mode, system_prompt, user_query) if run_ocr else ""
    return Document(
        id=document_id,
        source_path=str(path),
        title=path.stem,
        pages=[Page(id=f"{document_id}:page:1", number=1, text=text, image_path=image_path)],
        metadata={"source_type": "image", "ocr_run": run_ocr},
    )


def _document_from_pdf(
    path: Path,
    content: bytes,
    document_id: str,
    store: ArtifactStore,
    run_ocr: bool,
    mode: str,
    system_prompt: str,
    user_query: str,
    dpi: int,
) -> Optional[Document]:
    try:
        from pdf2image import convert_from_bytes, pdf2image
    except ImportError as exc:
        _logger.error("pdf2image not available: %s", exc)
        return None
    try:
        images = convert_from_bytes(content, dpi=dpi)
    except Exception as exc:
        _logger.error("Failed to convert PDF %s: %s", path.name, exc)
        return None
    pages = []
    for idx, image in enumerate(images, 1):
        try:
            image = image.convert("RGB")
        except Exception as exc:
            _logger.warning("Failed to convert PDF page %d: %s", idx, exc)
            continue
        image_path = _save_image_page(store, document_id, image, idx)
        text = _ocr_image(image, mode, system_prompt, user_query) if run_ocr else ""
        pages.append(Page(id=f"{document_id}:page:{idx}", number=idx, text=text, image_path=image_path))
    if not pages:
        _logger.warning("No pages extracted from PDF %s", path.name)
        return None
    return Document(
        id=document_id,
        source_path=str(path),
        title=path.stem,
        pages=pages,
        metadata={"source_type": "pdf", "ocr_run": run_ocr, "dpi": dpi},
    )


def ingest_path(
    path: str,
    store: Optional[ArtifactStore] = None,
    run_ocr: bool = False,
    mode: str = "",
    system_prompt: Optional[str] = None,
    user_query: Optional[str] = None,
    dpi: int = 0,
) -> Document:
    settings = get_settings()
    mode = mode or settings.default_ocr_mode
    dpi = dpi or settings.default_dpi
    store = store or ArtifactStore()
    source = Path(path)

    try:
        content = source.read_bytes()
    except (OSError, PermissionError) as exc:
        raise ValueError(f"Cannot read file {source}: {exc}") from exc

    document_id = stable_document_id(source, content)
    store.copy_raw(source, document_id)

    suffix = source.suffix.lower()
    document: Optional[Document] = None

    try:
        if suffix in {".txt", ".md", ".markdown"}:
            document = _document_from_text(source, content.decode("utf-8", errors="replace"), document_id)
        elif suffix == ".json":
            document = _document_from_history(source, document_id)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            document = _document_from_image(source, content, document_id, store, run_ocr, mode, system_prompt or "", user_query or "")
        elif suffix == ".pdf":
            document = _document_from_pdf(source, content, document_id, store, run_ocr, mode, system_prompt or "", user_query or "", dpi)
        else:
            raise ValueError(f"Unsupported input type: {suffix or source.name}")
    except Exception as exc:
        _logger.error("Ingestion failed for %s: %s", source.name, exc)
        raise

    if document is None:
        raise ValueError(f"Failed to ingest {source.name} — could not parse as {suffix or 'unknown'}")

    store.save_document(document)
    store.save_document_profile(build_document_profile(document))
    _logger.info("Ingested %s → %s (%d pages)", source.name, document.id, len(document.pages))
    return document
