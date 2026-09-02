"""
Multi-format, READ-ONLY content extractors for the local knowledge index.

Clean interface so new formats can be added later:

    class MyExtractor(BaseExtractor):
        extensions = {".xyz"}
        def extract(self, path: Path) -> Extracted:
            ...

Extraction never writes to disk. Corrupted / encrypted / password-protected
files degrade gracefully to status="failed" (metadata is still recorded).

Sections carry source locators ("page 3", "sheet Budget", "line 10-40")
so citations can reference the exact location in the original file.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Extraction statuses
STATUS_EXTRACTED = "extracted"
STATUS_METADATA_ONLY = "metadata_only"     # binary/unsupported: metadata kept
STATUS_FAILED = "failed"                    # corrupted/encrypted/timeout
STATUS_UNSUPPORTED = "unsupported"


@dataclass
class Section:
    """A piece of extracted text with its location in the source file."""
    locator: str          # e.g. "page 2", "sheet Budget", "line 10-40", ""
    text: str


@dataclass
class Extracted:
    """Result of extracting one file."""
    status: str
    sections: List[Section] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str = ""


class BaseExtractor:
    """Interface for format-specific extractors. READ-ONLY only."""
    extensions: Tuple[str, ...] = ()

    def can_handle(self, ext: str) -> bool:
        return ext.lower() in self.extensions

    def extract(self, path: Path) -> Extracted:
        raise NotImplementedError


# ── Plain text family (txt / md / code / config / logs) ───────────

_TEXT_CODE_EXT = {
    ".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".conf", ".toml",
    ".yaml", ".yml", ".properties", ".py", ".js", ".ts", ".java", ".c",
    ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh",
    ".bash", ".sql", ".r", ".kt", ".swift", ".lua", ".pl", ".html",
    ".htm", ".css", ".scss", ".dockerfile", ".makefile", ".cmake",
}


def _read_text_limited(path: Path, max_bytes: int) -> str:
    """Read a text file honoring a byte limit. READ-ONLY."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read(min(size, max_bytes))
    return data.decode("utf-8", errors="replace")


class TextExtractor(BaseExtractor):
    """TXT / MD / source code / config / logs — line-window sections."""
    extensions = tuple(_TEXT_CODE_EXT)

    def extract(self, path: Path) -> Extracted:
        try:
            text = _read_text_limited(path, 4 * 1024 * 1024)
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e))
        if "\x00" in text[:4096]:
            return Extracted(status=STATUS_METADATA_ONLY,
                             error="binary content detected")
        sections: List[Section] = []
        lines = text.splitlines()
        window = 80
        for i in range(0, len(lines), window):
            chunk = "\n".join(lines[i:i + window]).strip()
            if chunk:
                sections.append(Section(
                    locator=f"lines {i + 1}-{min(i + window, len(lines))}",
                    text=chunk))
        if not sections and text.strip():
            sections.append(Section(locator="", text=text.strip()))
        return Extracted(status=STATUS_EXTRACTED, sections=sections,
                         meta={"format": "text"})


class MarkdownExtractor(TextExtractor):
    """Markdown: split on headings for better section locators."""
    extensions = (".md", ".markdown")

    def extract(self, path: Path) -> Extracted:
        try:
            text = _read_text_limited(path, 4 * 1024 * 1024)
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e))
        sections: List[Section] = []
        current_title = "preamble"
        buf: List[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if buf:
                    body = "\n".join(buf).strip()
                    if body:
                        sections.append(Section(locator=current_title, text=body))
                    buf = []
                current_title = line.lstrip("#").strip() or "section"
            else:
                buf.append(line)
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(locator=current_title, text=body))
        if not sections and text.strip():
            sections.append(Section(locator="", text=text.strip()))
        return Extracted(status=STATUS_EXTRACTED, sections=sections,
                         meta={"format": "markdown"})


# ── PDF ───────────────────────────────────────────────────────────

class PdfExtractor(BaseExtractor):
    """PDF text extraction (per-page locators). READ-ONLY."""
    extensions = (".pdf",)

    def extract(self, path: Path) -> Extracted:
        sections: List[Section] = []
        meta: dict = {"format": "pdf"}
        # Try pypdf first, then PyPDF2, then pdfminer (all read-only).
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                return Extracted(status=STATUS_FAILED,
                                 error="password-protected PDF",
                                 meta=meta)
            meta["pages"] = len(reader.pages)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    txt = (page.extract_text() or "").strip()
                except Exception:
                    txt = ""
                if txt:
                    sections.append(Section(locator=f"page {i}", text=txt))
            if sections:
                return Extracted(status=STATUS_EXTRACTED,
                                 sections=sections, meta=meta)
            # No text layer — likely scanned. Metadata only (OCR optional
            # via the existing vision OCR pipeline; kept off by default
            # to bound indexing cost).
            return Extracted(status=STATUS_METADATA_ONLY,
                             error="no extractable text (scanned PDF?)",
                             meta=meta)
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e), meta=meta)


# ── DOCX ──────────────────────────────────────────────────────────

class DocxExtractor(BaseExtractor):
    extensions = (".docx",)

    def extract(self, path: Path) -> Extracted:
        try:
            import docx  # python-docx
            d = docx.Document(str(path))
            sections: List[Section] = []
            title = "body"
            buf: List[str] = []
            for para in d.paragraphs:
                style = getattr(para.style, "name", "") or ""
                text = (para.text or "").strip()
                if style.startswith("Heading") and text:
                    if buf:
                        body = "\n".join(buf).strip()
                        if body:
                            sections.append(Section(locator=title, text=body))
                        buf = []
                    title = text
                elif text:
                    buf.append(text)
            if buf:
                body = "\n".join(buf).strip()
                if body:
                    sections.append(Section(locator=title, text=body))
            if not sections:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="no text content", meta={"format": "docx"})
            return Extracted(status=STATUS_EXTRACTED, sections=sections,
                             meta={"format": "docx"})
        except Exception as e:
            # Encrypted/corrupted docx lands here
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "docx"})


# ── XLSX ──────────────────────────────────────────────────────────

class XlsxExtractor(BaseExtractor):
    extensions = (".xlsx", ".xlsm")

    def extract(self, path: Path) -> Extracted:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True,
                                        data_only=True)
            sections: List[Section] = []
            for ws in wb.worksheets:
                rows: List[str] = []
                for row in ws.iter_rows(values_only=True):
                    cells = ["" if c is None else str(c) for c in row]
                    if any(c.strip() for c in cells):
                        rows.append(" | ".join(cells))
                if rows:
                    sections.append(Section(
                        locator=f"sheet {ws.title}",
                        text="\n".join(rows[:5000])))
            wb.close()
            if not sections:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="empty workbook",
                                 meta={"format": "xlsx"})
            return Extracted(status=STATUS_EXTRACTED, sections=sections,
                             meta={"format": "xlsx"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "xlsx"})


# ── CSV ───────────────────────────────────────────────────────────

class CsvExtractor(BaseExtractor):
    extensions = (".csv", ".tsv")

    def extract(self, path: Path) -> Extracted:
        try:
            with open(path, "r", encoding="utf-8", errors="replace",
                      newline="") as f:
                sample = f.read(64 * 1024)
            dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
            reader = csv.reader(io.StringIO(sample), dialect=dialect)
            rows = [" | ".join(r) for r in reader if r]
            if not rows:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="empty csv", meta={"format": "csv"})
            return Extracted(
                status=STATUS_EXTRACTED,
                sections=[Section(locator="rows 1-%d" % len(rows),
                                  text="\n".join(rows))],
                meta={"format": "csv"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "csv"})


# ── JSON ──────────────────────────────────────────────────────────

class JsonExtractor(BaseExtractor):
    extensions = (".json",)

    def extract(self, path: Path) -> Extracted:
        try:
            text = _read_text_limited(path, 4 * 1024 * 1024)
            data = json.loads(text)
            pretty = json.dumps(data, indent=1, ensure_ascii=False,
                                default=str)
            return Extracted(
                status=STATUS_EXTRACTED,
                sections=[Section(locator="json", text=pretty[:200_000])],
                meta={"format": "json"})
        except json.JSONDecodeError as e:
            return Extracted(status=STATUS_FAILED,
                             error=f"invalid JSON: {e}",
                             meta={"format": "json"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "json"})


# ── XML ───────────────────────────────────────────────────────────

class XmlExtractor(BaseExtractor):
    extensions = (".xml",)

    def extract(self, path: Path) -> Extracted:
        try:
            tree = ET.parse(str(path))
            root = tree.getroot()
            parts: List[str] = []

            def walk(elem, depth=0):
                text = (elem.text or "").strip()
                if text:
                    parts.append("  " * depth + elem.tag + ": " + text)
                for child in elem:
                    if len(parts) < 20000:
                        walk(child, depth + 1)

            walk(root)
            if not parts:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="empty xml", meta={"format": "xml"})
            return Extracted(
                status=STATUS_EXTRACTED,
                sections=[Section(locator="xml", text="\n".join(parts))],
                meta={"format": "xml"})
        except ET.ParseError as e:
            return Extracted(status=STATUS_FAILED,
                             error=f"invalid XML: {e}",
                             meta={"format": "xml"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "xml"})


# ── HTML ──────────────────────────────────────────────────────────

class HtmlExtractor(BaseExtractor):
    extensions = (".html", ".htm")

    def extract(self, path: Path) -> Extracted:
        try:
            from bs4 import BeautifulSoup
            text = _read_text_limited(path, 4 * 1024 * 1024)
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = soup.title.get_text(strip=True) if soup.title else ""
            body = soup.get_text("\n").strip()
            body = re.sub(r"\n{3,}", "\n\n", body)
            if not body:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="no text content",
                                 meta={"format": "html"})
            locator = title or "html"
            return Extracted(
                status=STATUS_EXTRACTED,
                sections=[Section(locator=locator, text=body[:200_000])],
                meta={"format": "html"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "html"})


# ── PPTX ──────────────────────────────────────────────────────────

class PptxExtractor(BaseExtractor):
    extensions = (".pptx",)

    def extract(self, path: Path) -> Extracted:
        try:
            from pptx import Presentation
            prs = Presentation(str(path))
            sections: List[Section] = []
            for i, slide in enumerate(prs.slides, start=1):
                texts: List[str] = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        t = shape.text_frame.text.strip()
                        if t:
                            texts.append(t)
                if texts:
                    sections.append(Section(locator=f"slide {i}",
                                            text="\n".join(texts)))
            if not sections:
                return Extracted(status=STATUS_METADATA_ONLY,
                                 error="no text content",
                                 meta={"format": "pptx"})
            return Extracted(status=STATUS_EXTRACTED, sections=sections,
                             meta={"format": "pptx"})
        except Exception as e:
            return Extracted(status=STATUS_FAILED, error=str(e),
                             meta={"format": "pptx"})


# ── Registry ──────────────────────────────────────────────────────

_EXTRACTORS = [
    PdfExtractor(),
    DocxExtractor(),
    XlsxExtractor(),
    CsvExtractor(),
    JsonExtractor(),
    XmlExtractor(),
    HtmlExtractor(),
    PptxExtractor(),
    MarkdownExtractor(),
    TextExtractor(),   # last: broad text/code/config/log coverage
]


def get_extractor(ext: str) -> Optional[BaseExtractor]:
    """Find an extractor for a file extension (clean extension point)."""
    for ex in _EXTRACTORS:
        if ex.can_handle(ext):
            return ex
    return None


def supported_extensions() -> set:
    exts = set()
    for ex in _EXTRACTORS:
        exts.update(ex.extensions)
    return exts