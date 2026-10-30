"""Extract page text from every PDF in the configured source directory."""

from pathlib import Path
from typing import Any
import hashlib
import re

from pypdf import PdfReader

DEFAULT_DATA_DIR = "data"


def _is_reference_page(text: str) -> bool:
    """Exclude bibliographies and lists of other documents from clinical search."""
    head = text[:600].casefold()
    if re.search(r"(?im)^\s*(?:\d+[. ]*)?references\s*$", text[:600]):
        return True
    if "accompanying who documents" in head or "other who documents" in head:
        return True
    if "abbreviations and acronyms" in head or "table of contents" in head:
        return True
    numbered_citations = len(re.findall(r"(?m)^\s*\d{1,3}\.\s+[A-Z][^\n]{15,}", text))
    if numbered_citations >= 2 and re.search(r"doi:|doi\.org|et al\.|J Infect\.|Lancet|PubMed", text, re.I):
        return True
    links = len(re.findall(r"https?://|www\.", text, re.I))
    return links >= 3 and (links * 180 >= len(text) or links >= 6)


class PDFDirectoryRetriever:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def pdf_paths(self) -> list[Path]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"PDF directory not found: {self.data_dir}")
        paths = sorted(
            (path for path in self.data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.name.lower(),
        )
        if not paths:
            raise ValueError(f"No PDFs found in {self.data_dir}")
        return paths

    def fingerprint(self) -> str:
        """Hash source bytes so unchanged PDFs need no repeated text extraction."""
        digest = hashlib.sha256()
        for path in self.pdf_paths():
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def read_and_chunk(self) -> list[dict[str, Any]]:
        pages = []
        for path in self.pdf_paths():
            reader = PdfReader(path)
            in_references = False
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    if re.search(r"(?im)^\s*(?:annex|appendix)\b", text[:250]):
                        in_references = False
                    if re.search(r"(?im)^\s*(?:\d+[. ]*)?references\s*$", text):
                        in_references = True
                    references = in_references or _is_reference_page(text)
                    pages.append({
                        "source": path.name,
                        "page": page_number,
                        "text": text,
                        "section": "references" if references else "body",
                    })
        if not pages:
            raise ValueError(f"No extractable text found in PDFs in {self.data_dir}")
        return pages
