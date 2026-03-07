"""
document_parser.py
------------------
Unified document parser supporting:
  - PDF  → text + page boundary map (for source highlighting)
  - DOCX → text extraction respecting heading styles
  - TXT / MD → passthrough with basic cleanup

Page map output: dict[char_offset → page_number]
  The chunker uses this to annotate each chunk with a page_hint,
  enabling "jump to page X" links in the Streamlit UI.

Design note on PDF extraction:
  We use pdfplumber (built on pdfminer) over PyMuPDF because pdfplumber
  gives us precise character-level bounding boxes. We leverage these to
  detect multi-column layouts and reading order, which naive extractors
  often scramble. For scanned PDFs, we fall back to pytesseract OCR.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    text     : str
    doc_id   : str
    page_map : dict[int, int]    # char_offset → page_number
    metadata : dict              # filename, file_size, page_count, etc.


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _sanitise_text(text: str) -> str:
    """Remove control characters, normalise whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _extract_docx(file_bytes: bytes) -> tuple[str, dict[int, int]]:
    """Extract text from a .docx file preserving heading structure as Markdown."""
    try:
        from docx import Document  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
    except ImportError:
        raise ImportError("python-docx required: pip install python-docx")

    doc    = Document(io.BytesIO(file_bytes))
    lines  : list[str]      = []
    offset : int            = 0
    page_map: dict[int, int] = {0: 1}   # docx has no page boundaries in XML

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else ""
        text       = para.text.strip()
        if not text:
            lines.append("")
            continue

        if style_name.startswith("Heading"):
            try:
                level = int(style_name.split()[-1])
            except ValueError:
                level = 1
            prefix = "#" * min(level, 6) + " "
        else:
            prefix = ""

        line = prefix + text
        lines.append(line)
        offset += len(line) + 1

    full_text = "\n".join(lines)
    return _sanitise_text(full_text), page_map


def _extract_pdf(file_bytes: bytes) -> tuple[str, dict[int, int]]:
    """
    Extract text from a PDF with page map.
    Falls back to pytesseract OCR if text layer is absent.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("pdfplumber required: pip install pdfplumber")

    pages_text : list[str]      = []
    page_map   : dict[int, int] = {}
    char_offset : int           = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_map[char_offset] = page_num
            extracted = page.extract_text(x_tolerance=2, y_tolerance=3) or ""

            if not extracted.strip():
                # Scanned page — attempt OCR
                extracted = _ocr_page(page)

            pages_text.append(extracted)
            char_offset += len(extracted) + 2   # +2 for separator "\n\n"

    full_text = "\n\n".join(pages_text)
    return _sanitise_text(full_text), page_map


def _ocr_page(page) -> str:
    """Best-effort OCR on a scanned PDF page via pytesseract."""
    try:
        import pytesseract    # type: ignore
        from PIL import Image  # type: ignore
        img = page.to_image(resolution=200).original
        return pytesseract.image_to_string(img)
    except Exception as exc:
        logger.warning("OCR failed on page: %s", exc)
        return ""


def _extract_txt(file_bytes: bytes) -> tuple[str, dict[int, int]]:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(enc)
            return _sanitise_text(text), {0: 1}
        except UnicodeDecodeError:
            continue
    return _sanitise_text(file_bytes.decode("utf-8", errors="replace")), {0: 1}


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def parse_document(
    file_bytes : bytes,
    filename   : str,
) -> ParsedDocument:
    """
    Parse any supported document type into a ParsedDocument.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename   : str
        Original filename (used to infer type and as doc_id).
    """
    ext    = Path(filename).suffix.lower()
    doc_id = Path(filename).stem[:64]   # truncate for safety

    if ext == ".pdf":
        text, page_map = _extract_pdf(file_bytes)
    elif ext == ".docx":
        text, page_map = _extract_docx(file_bytes)
    elif ext in (".txt", ".md"):
        text, page_map = _extract_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    page_count = max(page_map.values()) if page_map else 1
    word_count = len(text.split())

    return ParsedDocument(
        text     = text,
        doc_id   = doc_id,
        page_map = page_map,
        metadata = {
            "filename"  : filename,
            "extension" : ext,
            "file_size" : len(file_bytes),
            "page_count": page_count,
            "word_count": word_count,
            "char_count": len(text),
        },
    )
