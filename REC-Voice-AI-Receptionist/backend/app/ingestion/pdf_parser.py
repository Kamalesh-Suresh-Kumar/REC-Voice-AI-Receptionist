from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass
class PDFDocument:
    path: str
    title: str
    pages: list[dict[str, Any]]


def extract_pdf(path: str | Path) -> PDFDocument:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    document = fitz.open(str(path))

    pages: list[dict[str, Any]] = []

    for page_number in range(document.page_count):
        page = document.load_page(page_number)

        # PyMuPDF typing can be ambiguous here.
        # For "text" extraction we explicitly treat the result as text.
        raw_text = page.get_text("text")
        text = str(raw_text).strip()

        if not text:
            continue

        pages.append(
            {
                "page": page_number + 1,
                "text": text,
            }
        )

    document.close()

    return PDFDocument(
        path=str(path),
        title=path.stem,
        pages=pages,
    )