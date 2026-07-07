"""CUDA-accelerated OCR: EasyOCR primary, PaddleOCR confidence-gated fallback.

Runs BEFORE RAGpack's own ingest path (embed_config.SETTINGS always forces
``ocr="never"``) so per-line OCR confidence -- which RAGpack's built-in docTR OCR
discards -- survives into the ingest report for ChunkLedger/Plumbline/Legigate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ragpack.extract import extract_text, is_garbled

_EASYOCR_READER = None
_PADDLE_OCR = None


def needs_ocr(pdf_path: str | Path) -> bool:
    """True if the PDF's text layer is empty/garbled -- reuses RAGpack's own check."""
    text = extract_text(pdf_path, ocr="never")
    return is_garbled(text)


@dataclass
class OcrResult:
    text: str
    engine: str
    mean_confidence: float
    per_page: list = field(default_factory=list)
    device: str = "unknown"


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _paddle_cuda_available() -> bool:
    try:
        import paddle

        return bool(paddle.device.is_compiled_with_cuda()) and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _rasterize(pdf_path: str | Path, dpi: int = 300) -> list:
    import fitz  # PyMuPDF

    pages = []
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def _easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=True)
    return _EASYOCR_READER


def _ocr_with_easyocr(pages: list) -> OcrResult:
    reader = _easyocr_reader()
    device = "cuda" if _torch_cuda_available() else "cpu"
    texts, confidences = [], []
    for png_bytes in pages:
        results = reader.readtext(png_bytes, detail=1)
        texts.append("\n".join(r[1] for r in results))
        page_conf = [float(r[2]) for r in results] or [0.0]
        confidences.append(sum(page_conf) / len(page_conf))
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OcrResult(text="\n\n".join(texts), engine="easyocr", mean_confidence=mean_conf,
                      per_page=confidences, device=device)


def _paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        from paddleocr import PaddleOCR

        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
    return _PADDLE_OCR


def _ocr_with_paddle(pages: list) -> OcrResult:
    ocr = _paddle_ocr()
    device = "cuda" if _paddle_cuda_available() else "cpu"
    texts, confidences = [], []
    for png_bytes in pages:
        result = ocr.ocr(png_bytes, cls=True)
        lines = (result[0] or []) if result else []
        texts.append("\n".join(line[1][0] for line in lines))
        page_conf = [float(line[1][1]) for line in lines] or [0.0]
        confidences.append(sum(page_conf) / len(page_conf))
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OcrResult(text="\n\n".join(texts), engine="paddleocr", mean_confidence=mean_conf,
                      per_page=confidences, device=device)


def ocr_pdf(pdf_path: str | Path, min_confidence: float = 0.55) -> OcrResult:
    """OCR a scanned/image PDF: EasyOCR first; if its mean confidence is below
    ``min_confidence``, retry with PaddleOCR and keep whichever scored higher.

    Either engine failing outright (CUDA OOM, a corrupt rasterized page, a missing
    model download) is treated as "that engine unavailable", not fatal -- only
    raises if BOTH engines fail."""
    pages = _rasterize(pdf_path)

    primary, primary_err = None, None
    try:
        primary = _ocr_with_easyocr(pages)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any OCR backend failure
        primary_err = exc

    if primary is not None and primary.mean_confidence >= min_confidence:
        return primary

    fallback, fallback_err = None, None
    try:
        fallback = _ocr_with_paddle(pages)
    except Exception as exc:  # noqa: BLE001
        fallback_err = exc

    if primary is None and fallback is None:
        raise RuntimeError(
            f"OCR failed for {pdf_path}: easyocr={primary_err!r} paddleocr={fallback_err!r}"
        )
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return fallback if fallback.mean_confidence > primary.mean_confidence else primary
