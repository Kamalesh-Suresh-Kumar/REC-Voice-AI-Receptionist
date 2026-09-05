from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image


# Windows Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


@dataclass
class PDFDocument:
    path: str
    title: str
    pages: list[dict[str, Any]]


def ocr_page(page) -> str:
    """
    Convert a scanned PDF page into an image
    and extract text using Tesseract OCR.
    """

    # Render page at approximately 300 DPI
    matrix = fitz.Matrix(300 / 72, 300 / 72)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    text = pytesseract.image_to_string(
        image,
        lang="eng",
    )

    return text.strip()


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

        # First try normal PDF text extraction
        raw_text = page.get_text("text")
        text = str(raw_text).strip()

        # If no text exists, use OCR
        if not text:
            print(
                f"[OCR] Scanned page detected: "
                f"{path.name} - page {page_number + 1}"
            )

            text = ocr_page(page)

            if text:
                print(
                    f"[OCR] Extracted text from "
                    f"{path.name} - page {page_number + 1}"
                )
            else:
                print(
                    f"[OCR] No text detected in "
                    f"{path.name} - page {page_number + 1}"
                )

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