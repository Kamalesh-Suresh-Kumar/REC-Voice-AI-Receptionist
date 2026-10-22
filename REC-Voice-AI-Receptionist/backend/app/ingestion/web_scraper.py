from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from io import BytesIO

import hashlib
import json
import re
import time

import requests
from bs4 import BeautifulSoup

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


# ============================================================
# WEB DOCUMENT
# ============================================================

@dataclass
class WebDocument:
    url: str
    title: str
    text: str
    sections: list[str]
    linked_documents: list[str]
    image_information: list[str]


# ============================================================
# REC WEBSITE SCRAPER
# ============================================================

class RECWebsiteScraper:

    def __init__(
        self,
        base_url: str,
        delay: float = 0.5,
        timeout: int = 25,
        max_pages: int | None = None,
        enable_ocr: bool = True,
        max_ocr_images_per_page: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.max_pages = max_pages

        self.enable_ocr = enable_ocr
        self.max_ocr_images_per_page = max_ocr_images_per_page

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "REC-AI-Receptionist-Research/1.0 "
                    "(educational project)"
                )
            }
        )

        self.allowed_domain = (
            urlparse(self.base_url)
            .netloc
            .lower()
        )

        # ----------------------------------------------------
        # Sitemap sources
        # ----------------------------------------------------

        self.sitemap_urls = [
            "https://aadhi.rajalakshmi.org/sitemap.xml",
            "https://www.rajalakshmi.org/sitemap.xml",
        ]

        # ----------------------------------------------------
        # Important REC pages
        #
        # These are crawl seeds only.
        # No answers are hard-coded here.
        # ----------------------------------------------------

        self.important_page_paths = [
            "/placement",
            "/placement-report",
            "/admissions",
            "/admission",
            "/academics",
            "/academic",
            "/programmes",
            "/programs",
            "/courses",
            "/facilities",
            "/library",
            "/central-library",
            "/sports",
            "/hostel",
            "/student-life",
            "/campus-life",
            "/administration",
            "/erp",
            "/contact",
            "/about",
            "/accreditation",
            "/iqac",
            "/naac",
            "/research",
        ]

        # ----------------------------------------------------
        # Department pages
        # ----------------------------------------------------

        self.department_urls = [
            "/departments/aero",
            "/departments/auto",
            "/departments/biomed",
            "/departments/biotech",
            "/departments/chem",
            "/departments/civil",
            "/departments/cse",
            "/departments/csecs",
            "/departments/csbs",
            "/departments/csd",
            "/departments/eee",
            "/departments/ece",
            "/departments/ft",
            "/departments/aids",
            "/departments/aiml",
            "/departments/mech",
            "/departments/mct",
            "/departments/ra",
            "/departments/it",
            "/departments/hands",
            "/departments/mba",
        ]

        # ----------------------------------------------------
        # Document extensions
        # ----------------------------------------------------

        self.document_extensions = {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".rar",
            ".7z",
        }

        # ----------------------------------------------------
        # Non-page extensions
        # ----------------------------------------------------

        self.skip_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".ico",
            ".mp3",
            ".wav",
            ".mp4",
            ".avi",
            ".mov",
            ".css",
            ".js",
            ".xml",
            ".json",
            ".map",
            ".woff",
            ".woff2",
            ".ttf",
        } | self.document_extensions

        # ----------------------------------------------------
        # Blocked URLs
        # ----------------------------------------------------

        self.blocked_keywords = {
            "/login",
            "/signin",
            "/signup",
            "/register",
            "/logout",
            "/wp-admin",
            "/api/",
            "/_next/",
        }

        # ----------------------------------------------------
        # Configure OCR
        # ----------------------------------------------------

        self.ocr_available = False

        self._configure_ocr()

    # ========================================================
    # OCR CONFIGURATION
    # ========================================================

    def _configure_ocr(self) -> None:

        if not self.enable_ocr:
            print("[OCR] OCR disabled.")
            return

        if pytesseract is None:
            print(
                "[OCR] pytesseract is not installed. "
                "OCR disabled."
            )
            return

        if Image is None:
            print(
                "[OCR] Pillow is not installed. "
                "OCR disabled."
            )
            return

        try:

            version = pytesseract.get_tesseract_version()

            print(
                f"[OCR] Tesseract detected: {version}"
            )

            self.ocr_available = True
            return

        except Exception:
            pass

        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Tesseract-OCR"
                / "tesseract.exe"
            ),
        ]

        for tesseract_path in windows_paths:

            path = Path(tesseract_path)

            if not path.exists():
                continue

            try:

                pytesseract.pytesseract.tesseract_cmd = (
                    str(path)
                )

                version = (
                    pytesseract.get_tesseract_version()
                )

                print(
                    f"[OCR] Tesseract detected: {version}"
                )

                self.ocr_available = True
                return

            except Exception as exc:

                print(
                    f"[OCR] Tesseract configuration "
                    f"failed: {exc}"
                )

        print(
            "[OCR] Tesseract executable not found. "
            "Image metadata will still be extracted."
        )

    # ========================================================
    # URL NORMALIZATION
    # ========================================================

    def _normalize_url(
        self,
        url: str,
        allow_documents: bool = False,
    ) -> str | None:

        if not url:
            return None

        url = url.strip()

        if url.startswith(
            (
                "mailto:",
                "tel:",
                "javascript:",
                "#",
                "data:",
            )
        ):
            return None

        absolute_url = urljoin(
            self.base_url + "/",
            url,
        )

        parsed = urlparse(
            absolute_url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return None

        hostname = (
            parsed.hostname or ""
        ).lower()

        if hostname != self.allowed_domain:
            return None

        path = parsed.path or "/"

        path = re.sub(
            r"/+",
            "/",
            path,
        )

        if path != "/":
            path = path.rstrip("/")

        extension = Path(
            path
        ).suffix.lower()

        if extension in self.document_extensions:

            if not allow_documents:
                return None

        elif extension in self.skip_extensions:

            return None

        lower_url = absolute_url.lower()

        for keyword in self.blocked_keywords:

            if keyword in lower_url:
                return None

        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                "",
                "",
                "",
            )
        )

    # ========================================================
    # DOMAIN CHECK
    # ========================================================

    def _is_same_domain(
        self,
        url: str,
    ) -> bool:

        return (
            urlparse(url)
            .netloc
            .lower()
            == self.allowed_domain
        )

    # ========================================================
    # SITEMAP URL CONVERSION
    # ========================================================

    def _convert_sitemap_url(
        self,
        url: str,
    ) -> str | None:

        parsed = urlparse(url)

        path = parsed.path or "/"

        official_url = urlunparse(
            (
                "https",
                self.allowed_domain,
                path,
                "",
                "",
                "",
            )
        )

        return self._normalize_url(
            official_url
        )

    # ========================================================
    # LOAD SITEMAP
    # ========================================================

    def load_sitemap_urls(
        self,
    ) -> list[str]:

        discovered: set[str] = set()

        for sitemap_url in self.sitemap_urls:

            print(
                f"[SITEMAP] Trying: {sitemap_url}"
            )

            try:

                response = self.session.get(
                    sitemap_url,
                    timeout=self.timeout,
                )

                response.raise_for_status()

            except requests.RequestException as exc:

                print(
                    f"[SITEMAP] Failed: {exc}"
                )

                continue

            soup = BeautifulSoup(
                response.text,
                "xml",
            )

            locations = soup.find_all(
                "loc"
            )

            for location in locations:

                raw_url = location.get_text(
                    strip=True
                )

                converted = (
                    self._convert_sitemap_url(
                        raw_url
                    )
                )

                if converted:
                    discovered.add(
                        converted
                    )

            if discovered:

                print(
                    f"[SITEMAP] Loaded "
                    f"{len(discovered)} URLs"
                )

                break

        return sorted(
            discovered
        )

    # ========================================================
    # DECODE JSON STRING
    # ========================================================

    def _decode_json_string(
        self,
        value: str,
    ) -> str:

        if not value:
            return ""

        try:

            decoded = json.loads(
                '"' + value + '"'
            )

            if isinstance(
                decoded,
                str,
            ):
                return decoded

        except Exception:
            pass

        value = (
            value
            .replace('\\"', '"')
            .replace("\\\\", "\\")
            .replace("\\/", "/")
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\t", " ")
        )

        return value

    # ========================================================
    # EXTRACT NEXT.JS TEXT VALUES
    # ========================================================

    def _extract_nextjs_text_values(
        self,
        html: str,
    ) -> list[str]:

        if not html:
            return []

        results: list[str] = []

        patterns = [
            r'\\"text\\":\\"(.*?)\\"',
            r'"text":"(.*?)"',
        ]

        for pattern in patterns:

            try:

                matches = re.findall(
                    pattern,
                    html,
                    flags=re.DOTALL,
                )

            except Exception:
                continue

            for value in matches:

                text = self._decode_json_string(
                    value
                )

                text = re.sub(
                    r"\s+",
                    " ",
                    text,
                ).strip()

                if not text:
                    continue

                if len(text) < 2:
                    continue

                if len(text) > 3000:
                    continue

                if text not in results:
                    results.append(text)

        return results

    # ========================================================
    # NEXT.JS FACULTY EXTRACTION
    # ========================================================

    def _extract_nextjs_faculty(
        self,
        html: str,
    ) -> list[str]:

        faculty_texts: list[str] = []

        if not html:
            return []

        faculty_markers = [
            '\\"faculty\\":{',
            '"faculty":{',
        ]

        faculty_start = -1
        faculty_marker = ""

        for marker in faculty_markers:

            position = html.find(
                marker
            )

            if position != -1:

                faculty_start = position
                faculty_marker = marker
                break

        if faculty_start == -1:
            return []

        end_markers = [
            '\\"tableData\\"',
            '"tableData"',
        ]

        faculty_end = -1

        for marker in end_markers:

            position = html.find(
                marker,
                faculty_start + len(
                    faculty_marker
                ),
            )

            if position != -1:

                faculty_end = position
                break

        if faculty_end == -1:

            faculty_end = min(
                len(html),
                faculty_start + 150000,
            )

        faculty_block = html[
            faculty_start:faculty_end
        ]

        patterns = [
            re.compile(
                r'\\"text\\":\\"(.*?)\\"',
                re.DOTALL,
            ),
            re.compile(
                r'"text":"(.*?)"',
                re.DOTALL,
            ),
        ]

        faculty_keywords = [
            "professor",
            "prof.",
            "associate professor",
            "assistant professor",
            "hod",
            "head",
            "dean",
            "principal",
            "lecturer",
            "director",
        ]

        for pattern in patterns:

            matches = pattern.findall(
                faculty_block
            )

            for value in matches:

                text = self._decode_json_string(
                    value
                )

                text = re.sub(
                    r"\s+",
                    " ",
                    text,
                ).strip()

                if not text:
                    continue

                if not any(
                    keyword in text.lower()
                    for keyword in faculty_keywords
                ):
                    continue

                if text not in faculty_texts:

                    faculty_texts.append(
                        text
                    )

        return faculty_texts

    # ========================================================
    # EXTRACT NEXT.JS INSTITUTIONAL CONTENT
    # ========================================================

    def _extract_nextjs_institutional_content(
        self,
        html: str,
    ) -> list[str]:

        """
        Extract useful textual values from Next.js
        serialized data.

        This does not create or invent institutional
        information. It only preserves text already
        present in the official page response.
        """

        if not html:
            return []

        values = (
            self._extract_nextjs_text_values(
                html
            )
        )

        useful: list[str] = []

        ignore_values = {
            "normal",
            "left",
            "right",
            "center",
            "justify",
            "inherit",
            "bold",
            "italic",
        }

        for value in values:

            lower = value.lower().strip()

            if lower in ignore_values:
                continue

            # Ignore obvious UI/internal values
            if lower.startswith(
                (
                    "http://",
                    "https://",
                    "data:image",
                )
            ):
                continue

            if "className" in value:
                continue

            if "undefined" == lower:
                continue

            if value not in useful:
                useful.append(value)

        return useful

    # ========================================================
    # IMAGE SOURCE EXTRACTION
    # ========================================================

    def _get_image_source(
        self,
        image,
    ) -> str | None:

        attributes = [
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
            "data-image",
            "data-url",
        ]

        for attribute in attributes:

            value = image.get(
                attribute
            )

            if isinstance(
                value,
                str,
            ) and value.strip():

                return value.strip()

        srcset_attributes = [
            "srcset",
            "data-srcset",
        ]

        for attribute in srcset_attributes:

            value = image.get(
                attribute
            )

            if not isinstance(
                value,
                str,
            ):
                continue

            candidates = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

            if candidates:

                last_candidate = candidates[-1]

                image_url = (
                    last_candidate
                    .split()[0]
                    .strip()
                )

                if image_url:
                    return image_url

        return None

    # ========================================================
    # NEARBY IMAGE TEXT
    # ========================================================

    def _get_nearby_image_text(
        self,
        image,
    ) -> str:

        texts: list[str] = []

        parent = image.parent

        if parent is not None:

            caption = parent.find(
                "figcaption"
            )

            if caption is not None:

                caption_text = caption.get_text(
                    " ",
                    strip=True,
                )

                if caption_text:
                    texts.append(
                        caption_text
                    )

        current = image.parent

        for _ in range(3):

            if current is None:
                break

            parent_text = current.get_text(
                " ",
                strip=True,
            )

            parent_text = re.sub(
                r"\s+",
                " ",
                parent_text,
            ).strip()

            if (
                parent_text
                and len(parent_text) <= 1500
            ):
                texts.append(
                    parent_text
                )

            current = current.parent

        previous_heading = image.find_previous(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ]
        )

        if previous_heading is not None:

            heading_text = (
                previous_heading.get_text(
                    " ",
                    strip=True,
                )
            )

            if heading_text:
                texts.append(
                    heading_text
                )

        unique_texts = []

        seen = set()

        for text in texts:

            normalized = " ".join(
                text.split()
            )

            if (
                not normalized
                or normalized in seen
            ):
                continue

            seen.add(
                normalized
            )

            unique_texts.append(
                normalized
            )

        return " | ".join(
            unique_texts[:5]
        )

    # ========================================================
    # OCR IMAGE
    # ========================================================

    def _ocr_image(
        self,
        image_url: str,
    ) -> str:

        if not self.ocr_available:
            return ""

        try:

            response = self.session.get(
                image_url,
                timeout=self.timeout,
            )

            response.raise_for_status()

            content_type = (
                response.headers.get(
                    "Content-Type",
                    "",
                )
                .lower()
            )

            if (
                content_type
                and "image" not in content_type
            ):
                return ""

            if Image is None:
                return ""

            image_module: Any = Image

            image = image_module.open(
                BytesIO(
                    response.content
                )
            )

            width, height = image.size

            if width < 80 or height < 30:
                return ""

            if image.mode not in (
                "RGB",
                "L",
            ):
                image = image.convert(
                    "RGB"
                )

            # Enlarge small images for OCR
            if width < 1400:

                scale = min(
                    4.0,
                    1400 / max(width, 1),
                )

                new_width = int(
                    width * scale
                )

                new_height = int(
                    height * scale
                )

                image = image.resize(
                    (
                        new_width,
                        new_height,
                    )
                )

            if pytesseract is None:
                return ""

            ocr_engine: Any = pytesseract

            text = ocr_engine.image_to_string(
                image,
                config="--psm 6",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if len(text) < 3:
                return ""

            return text

        except Exception as exc:

            print(
                f"        [OCR WARNING] "
                f"{image_url}: {exc}"
            )

            return ""

    # ========================================================
    # EXTRACT IMAGE INFORMATION
    # ========================================================

    def _extract_image_information(
        self,
        main_content,
        page_url: str,
    ) -> list[str]:

        image_information: list[str] = []

        images = main_content.find_all(
            "img"
        )

        if not images:
            return []

        print(
            f"    [IMAGES] Found "
            f"{len(images)} images"
        )

        ocr_count = 0

        for index, image in enumerate(
            images,
            start=1,
        ):

            raw_src = self._get_image_source(
                image
            )

            if not raw_src:
                continue

            image_url = urljoin(
                page_url,
                raw_src,
            )

            parsed = urlparse(
                image_url
            )

            if parsed.scheme not in {
                "http",
                "https",
            }:
                continue

            alt_text = image.get(
                "alt",
                "",
            )

            title_text = image.get(
                "title",
                "",
            )

            if not isinstance(
                alt_text,
                str,
            ):
                alt_text = ""

            if not isinstance(
                title_text,
                str,
            ):
                title_text = ""

            alt_text = re.sub(
                r"\s+",
                " ",
                alt_text,
            ).strip()

            title_text = re.sub(
                r"\s+",
                " ",
                title_text,
            ).strip()

            caption_text = ""

            parent = image.parent

            if parent is not None:

                caption = parent.find(
                    "figcaption"
                )

                if caption is not None:

                    caption_text = (
                        caption.get_text(
                            " ",
                            strip=True,
                        )
                    )

                    caption_text = re.sub(
                        r"\s+",
                        " ",
                        caption_text,
                    ).strip()

            nearby_text = (
                self._get_nearby_image_text(
                    image
                )
            )

            image_link = ""

            anchor = image.find_parent(
                "a"
            )

            if anchor is not None:

                href = anchor.get(
                    "href"
                )

                if isinstance(
                    href,
                    str,
                ):

                    image_link = urljoin(
                        page_url,
                        href,
                    )

            ocr_text = ""

            if (
                self.ocr_available
                and ocr_count
                < self.max_ocr_images_per_page
            ):

                print(
                    f"        [OCR] Image {index}"
                )

                ocr_text = self._ocr_image(
                    image_url
                )

                ocr_count += 1

            useful_information = (
                alt_text
                or title_text
                or caption_text
                or nearby_text
                or ocr_text
            )

            if not useful_information:
                continue

            information = [
                f"IMAGE {index}",
                f"IMAGE_URL: {image_url}",
            ]

            if image_link:
                information.append(
                    f"IMAGE_LINK: {image_link}"
                )

            if alt_text:
                information.append(
                    f"ALT_TEXT: {alt_text}"
                )

            if title_text:
                information.append(
                    f"TITLE_TEXT: {title_text}"
                )

            if caption_text:
                information.append(
                    f"CAPTION: {caption_text}"
                )

            if nearby_text:
                information.append(
                    f"NEARBY_TEXT: {nearby_text}"
                )

            if ocr_text:
                information.append(
                    f"OCR_TEXT: {ocr_text}"
                )

            image_information.append(
                "\n".join(
                    information
                )
            )

        print(
            f"    [IMAGES] Useful image information: "
            f"{len(image_information)}"
        )

        return image_information

    # ========================================================
    # EXTRACT LINK TEXT
    # ========================================================

    def _extract_link_text(
        self,
        anchor,
    ) -> str:

        text = anchor.get_text(
            " ",
            strip=True,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # ========================================================
    # CLEAN PAGE
    # ========================================================

    def _clean_page(
        self,
        soup: BeautifulSoup,
        page_url: str,
    ) -> tuple[
        str,
        list[str],
        list[str],
        list[str],
    ]:

        # ----------------------------------------------------
        # Remove scripts AFTER raw HTML has been processed.
        # ----------------------------------------------------

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "canvas",
                "form",
                "header",
                "footer",
                "nav",
            ]
        ):
            element.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.select_one(
                '[role="main"]'
            )
            or soup.body
        )

        if main_content is None:
            return "", [], [], []

        # ----------------------------------------------------
        # Remove navigation/UI elements
        # ----------------------------------------------------

        for selector in [
            ".navbar",
            ".navigation",
            ".menu",
            ".breadcrumb",
            ".breadcrumbs",
            ".cookie",
            ".cookie-banner",
            ".modal",
            ".popup",
            ".social-share",
            ".footer",
            ".header",
        ]:

            for element in main_content.select(
                selector
            ):
                element.decompose()

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        image_information = (
            self._extract_image_information(
                main_content,
                page_url,
            )
        )

        # ----------------------------------------------------
        # Linked documents
        # ----------------------------------------------------

        linked_documents: set[str] = set()

        for anchor in main_content.find_all(
            "a",
            href=True,
        ):

            href = anchor.get(
                "href"
            )

            if not isinstance(
                href,
                str,
            ):
                continue

            absolute = urljoin(
                page_url,
                href,
            )

            parsed = urlparse(
                absolute
            )

            extension = Path(
                parsed.path
            ).suffix.lower()

            if extension in self.document_extensions:

                document_url = (
                    self._normalize_url(
                        absolute,
                        allow_documents=True,
                    )
                )

                if document_url:

                    linked_documents.add(
                        document_url
                    )

        # ----------------------------------------------------
        # Preserve headings, paragraphs, lists and tables
        # ----------------------------------------------------

        blocks: list[str] = []
        sections: list[str] = []

        heading_tags = {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }

        current_section = ""

        elements = main_content.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "li",
                "table",
                "blockquote",
                "dt",
                "dd",
            ]
        )

        for element in elements:

            tag_name = (
                element.name or ""
            ).lower()

            raw_text = element.get_text(
                " ",
                strip=True,
            )

            raw_text = re.sub(
                r"\s+",
                " ",
                raw_text,
            ).strip()

            if not raw_text:
                continue

            if len(raw_text) < 2:
                continue

            # ------------------------------------------------
            # Heading
            # ------------------------------------------------

            if tag_name in heading_tags:

                current_section = raw_text

                if current_section not in sections:

                    sections.append(
                        current_section
                    )

                blocks.append(
                    f"SECTION: {current_section}"
                )

                continue

            # ------------------------------------------------
            # Table
            # ------------------------------------------------

            if tag_name == "table":

                rows = []

                for row in element.find_all(
                    "tr"
                ):

                    cells = []

                    for cell in row.find_all(
                        [
                            "th",
                            "td",
                        ]
                    ):

                        cell_text = cell.get_text(
                            " ",
                            strip=True,
                        )

                        cell_text = re.sub(
                            r"\s+",
                            " ",
                            cell_text,
                        ).strip()

                        if cell_text:
                            cells.append(
                                cell_text
                            )

                    if cells:

                        rows.append(
                            " | ".join(
                                cells
                            )
                        )

                if rows:

                    table_text = (
                        f"TABLE"
                        + (
                            f" | SECTION: "
                            f"{current_section}"
                            if current_section
                            else ""
                        )
                        + "\n"
                        + "\n".join(rows)
                    )

                    blocks.append(
                        table_text
                    )

                continue

            # ------------------------------------------------
            # Normal content
            # ------------------------------------------------

            if current_section:

                blocks.append(
                    f"{raw_text}"
                )

            else:

                blocks.append(
                    raw_text
                )

        # ----------------------------------------------------
        # Extract useful link labels.
        #
        # This helps preserve navigation information such as:
        # Admissions, Library, ERP, Placement, Sports, etc.
        # ----------------------------------------------------

        link_information = []

        for anchor in main_content.find_all(
            "a",
            href=True,
        ):

            label = self._extract_link_text(
                anchor
            )

            href = anchor.get(
                "href"
            )

            if (
                not label
                or not isinstance(href, str)
            ):
                continue

            if len(label) > 200:
                continue

            absolute = urljoin(
                page_url,
                href,
            )

            normalized = self._normalize_url(
                absolute
            )

            if normalized is None:
                continue

            line = (
                f"LINK: {label} -> {normalized}"
            )

            if line not in link_information:
                link_information.append(
                    line
                )

        if link_information:

            blocks.append(
                "PAGE LINKS\n"
                + "\n".join(
                    link_information
                )
            )

        # ----------------------------------------------------
        # Fallback text
        # ----------------------------------------------------

        if len(blocks) < 3:

            text = main_content.get_text(
                separator="\n"
            )

            blocks = []

            for line in text.splitlines():

                line = re.sub(
                    r"\s+",
                    " ",
                    line,
                ).strip()

                if line:
                    blocks.append(
                        line
                    )

        # ----------------------------------------------------
        # Remove duplicate blocks
        # ----------------------------------------------------

        cleaned_blocks = []

        previous = None

        for block in blocks:

            block = block.strip()

            if not block:
                continue

            if block == previous:
                continue

            cleaned_blocks.append(
                block
            )

            previous = block

        text = "\n".join(
            cleaned_blocks
        ).strip()

        return (
            text,
            sections,
            sorted(
                linked_documents
            ),
            image_information,
        )

    # ========================================================
    # FETCH PAGE
    # ========================================================

    def fetch(
        self,
        url: str,
    ) -> WebDocument | None:

        normalized_url = (
            self._normalize_url(
                url
            )
        )

        if normalized_url is None:
            return None

        try:

            response = self.session.get(
                normalized_url,
                timeout=self.timeout,
                allow_redirects=True,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            print(
                f"[ERROR] "
                f"{normalized_url}: {exc}"
            )

            return None

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "text/html" not in content_type:

            print(
                f"[SKIP] Non-HTML: "
                f"{normalized_url}"
            )

            return None

        # ----------------------------------------------------
        # KEEP RAW HTML
        # ----------------------------------------------------

        raw_html = response.text

        # ----------------------------------------------------
        # FACULTY
        # ----------------------------------------------------

        faculty_information = (
            self._extract_nextjs_faculty(
                raw_html
            )
        )

        if faculty_information:

            print(
                f"    [FACULTY] Found "
                f"{len(faculty_information)} "
                f"faculty entries"
            )

        # ----------------------------------------------------
        # NEXT.JS CONTENT
        # ----------------------------------------------------

        nextjs_content = (
            self._extract_nextjs_institutional_content(
                raw_html
            )
        )

        print(
            f"    [NEXT.JS] Extracted "
            f"{len(nextjs_content)} "
            f"serialized text values"
        )

        # ----------------------------------------------------
        # BEAUTIFULSOUP
        # ----------------------------------------------------

        soup = BeautifulSoup(
            raw_html,
            "lxml",
        )

        title = ""

        if soup.title:

            title = soup.title.get_text(
                strip=True
            )

        (
            text,
            sections,
            linked_documents,
            image_information,
        ) = self._clean_page(
            soup,
            normalized_url,
        )

        # ----------------------------------------------------
        # Add faculty
        # ----------------------------------------------------

        if faculty_information:

            faculty_section = (
                "\n\n"
                "FACULTY\n"
                "=======\n"
                + "\n".join(
                    faculty_information
                )
                + "\n"
            )

            text = (
                text.rstrip()
                + faculty_section
            )

            if "FACULTY" not in sections:

                sections.append(
                    "FACULTY"
                )

        # ----------------------------------------------------
        # Add useful Next.js serialized values.
        #
        # Keep only values that are not already present
        # in normal page extraction.
        # ----------------------------------------------------

        if nextjs_content:

            existing_text = text.lower()

            unique_nextjs = []

            for value in nextjs_content:

                normalized_value = (
                    re.sub(
                        r"\s+",
                        " ",
                        value,
                    ).strip()
                )

                if not normalized_value:
                    continue

                # Avoid dumping huge duplicated content
                if (
                    normalized_value.lower()
                    in existing_text
                ):
                    continue

                if normalized_value not in unique_nextjs:

                    unique_nextjs.append(
                        normalized_value
                    )

            if unique_nextjs:

                text += (
                    "\n\n"
                    "NEXTJS-SERIALIZED-CONTENT\n"
                    "==========================\n"
                    + "\n".join(
                        unique_nextjs
                    )
                    + "\n"
                )

                if (
                    "NEXTJS-SERIALIZED-CONTENT"
                    not in sections
                ):

                    sections.append(
                        "NEXTJS-SERIALIZED-CONTENT"
                    )

        if not text and not image_information:
            return None

        return WebDocument(
            url=normalized_url,
            title=title,
            text=text,
            sections=sections,
            linked_documents=linked_documents,
            image_information=image_information,
        )

    # ========================================================
    # DISCOVER LINKS
    # ========================================================

    def discover_links(
        self,
        start_url: str,
    ) -> tuple[
        list[str],
        list[str],
    ]:

        try:

            response = self.session.get(
                start_url,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            print(
                f"[ERROR] "
                f"Link discovery failed: "
                f"{start_url}: {exc}"
            )

            return [], []

        raw_html = response.text

        soup = BeautifulSoup(
            raw_html,
            "lxml",
        )

        internal_links: set[str] = set()
        document_links: set[str] = set()

        # ----------------------------------------------------
        # Normal HTML links
        # ----------------------------------------------------

        for anchor in soup.find_all(
            "a",
            href=True,
        ):

            href = anchor.get(
                "href"
            )

            if not isinstance(
                href,
                str,
            ):
                continue

            absolute = urljoin(
                start_url,
                href,
            )

            parsed = urlparse(
                absolute
            )

            if parsed.scheme not in {
                "http",
                "https",
            }:
                continue

            extension = Path(
                parsed.path
            ).suffix.lower()

            if extension in self.document_extensions:

                document_url = (
                    self._normalize_url(
                        absolute,
                        allow_documents=True,
                    )
                )

                if document_url:

                    document_links.add(
                        document_url
                    )

                continue

            normalized_url = (
                self._normalize_url(
                    absolute
                )
            )

            if normalized_url is None:
                continue

            if not self._is_same_domain(
                normalized_url
            ):
                continue

            internal_links.add(
                normalized_url
            )

        # ----------------------------------------------------
        # Next.js serialized URLs
        #
        # This catches URLs present in page JSON that
        # aren't rendered as ordinary <a> tags.
        # ----------------------------------------------------

        url_patterns = [
            r'https?:\\?/\\?/www\.rajalakshmi\.org[^\s"\\]+',
            r'https?:\\?/\\?/rajalakshmi\.org[^\s"\\]+',
        ]

        for pattern in url_patterns:

            try:

                matches = re.findall(
                    pattern,
                    raw_html,
                    flags=re.IGNORECASE,
                )

            except Exception:
                matches = []

            for raw_url in matches:

                raw_url = (
                    raw_url
                    .replace("\\/", "/")
                    .replace("\\u002F", "/")
                    .replace("\\u0026", "&")
                )

                normalized_url = (
                    self._normalize_url(
                        raw_url
                    )
                )

                if normalized_url:

                    internal_links.add(
                        normalized_url
                    )

        return (
            sorted(
                internal_links
            ),
            sorted(
                document_links
            ),
        )

    # ========================================================
    # CRAWL WEBSITE
    # ========================================================

    def crawl(
        self,
        start_url: str | None = None,
    ) -> tuple[
        list[WebDocument],
        list[str],
    ]:

        queue: list[str] = []
        queued: set[str] = set()

        def add_seed(
            url: str | None,
        ) -> None:

            if not url:
                return

            normalized = (
                self._normalize_url(
                    url
                )
            )

            if (
                normalized
                and normalized not in queued
            ):

                queue.append(
                    normalized
                )

                queued.add(
                    normalized
                )

        # ----------------------------------------------------
        # Homepage
        # ----------------------------------------------------

        add_seed(
            start_url
            or self.base_url
        )

        # ----------------------------------------------------
        # Sitemap
        # ----------------------------------------------------

        for url in self.load_sitemap_urls():

            add_seed(
                url
            )

        # ----------------------------------------------------
        # Important pages
        # ----------------------------------------------------

        for important_path in (
            self.important_page_paths
        ):

            add_seed(
                urljoin(
                    self.base_url + "/",
                    important_path,
                )
            )

        # ----------------------------------------------------
        # Departments
        # ----------------------------------------------------

        for department_path in (
            self.department_urls
        ):

            add_seed(
                urljoin(
                    self.base_url + "/",
                    department_path,
                )
            )

        visited: set[str] = set()

        documents: list[
            WebDocument
        ] = []

        linked_documents: set[str] = set()

        while queue:

            if (
                self.max_pages is not None
                and len(documents)
                >= self.max_pages
            ):

                print(
                    "[LIMIT] "
                    "Maximum page limit reached."
                )

                break

            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            print()
            print(
                f"[{len(documents) + 1}] "
                f"Fetching: {url}"
            )

            document = self.fetch(
                url
            )

            # ------------------------------------------------
            # Discover links
            # ------------------------------------------------

            try:

                (
                    links,
                    page_documents,
                ) = self.discover_links(
                    url
                )

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Link discovery failed: "
                    f"{exc}"
                )

                links = []
                page_documents = []

            for document_url in (
                page_documents
            ):

                linked_documents.add(
                    document_url
                )

            # ------------------------------------------------
            # Save document
            # ------------------------------------------------

            if document is not None:

                word_count = len(
                    document.text.split()
                )

                print(
                    f"    Title: "
                    f"{document.title}"
                )

                print(
                    f"    Words: "
                    f"{word_count}"
                )

                print(
                    f"    Sections: "
                    f"{len(document.sections)}"
                )

                print(
                    f"    Linked documents: "
                    f"{len(document.linked_documents)}"
                )

                print(
                    f"    Image information: "
                    f"{len(document.image_information)}"
                )

                for document_url in (
                    document.linked_documents
                ):

                    linked_documents.add(
                        document_url
                    )

                if (
                    word_count >= 20
                    or document.image_information
                ):

                    documents.append(
                        document
                    )

                    print(
                        "    [SAVED]"
                    )

                else:

                    print(
                        "    [SKIP] "
                        "Insufficient text"
                    )

            # ------------------------------------------------
            # Add discovered pages
            # ------------------------------------------------

            new_links = 0

            for link in links:

                if (
                    link not in visited
                    and link not in queued
                ):

                    queue.append(
                        link
                    )

                    queued.add(
                        link
                    )

                    new_links += 1

            print(
                f"    New HTML links: "
                f"{new_links}"
            )

            time.sleep(
                self.delay
            )

        return (
            documents,
            sorted(
                linked_documents
            ),
        )

    # ========================================================
    # SAVE DOCUMENTS
    # ========================================================

    def save_documents(
        self,
        documents: list[WebDocument],
        linked_documents: list[str],
        output_dir: str | Path,
    ) -> None:

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = []

        # ----------------------------------------------------
        # Save pages
        # ----------------------------------------------------

        for index, document in enumerate(
            documents,
            start=1,
        ):

            url_hash = hashlib.md5(
                document.url.encode(
                    "utf-8"
                )
            ).hexdigest()[:8]

            filename = (
                f"web_{index:03d}_"
                f"{url_hash}.txt"
            )

            file_path = (
                output_path
                / filename
            )

            image_section = ""

            if document.image_information:

                image_section = (
                    "\n\n"
                    "IMAGE-BASED INFORMATION\n"
                    "========================\n\n"
                    + "\n\n".join(
                        document.image_information
                    )
                    + "\n"
                )

            content = (
                f"TITLE: "
                f"{document.title}\n"
                f"SOURCE_URL: "
                f"{document.url}\n"
                f"SOURCE_TYPE: website\n"
                f"SECTIONS: "
                f"{json.dumps(document.sections, ensure_ascii=False)}\n"
                f"\n"
                f"{document.text}\n"
                f"{image_section}"
            )

            file_path.write_text(
                content,
                encoding="utf-8",
            )

            manifest.append(
                {
                    "file": filename,
                    "title": document.title,
                    "url": document.url,
                    "source_type": "website",
                    "word_count": len(
                        document.text.split()
                    ),
                    "section_count": len(
                        document.sections
                    ),
                    "sections": document.sections,
                    "linked_documents": (
                        document.linked_documents
                    ),
                    "image_count": len(
                        document.image_information
                    ),
                    "image_information": (
                        document.image_information
                    ),
                }
            )

        # ----------------------------------------------------
        # Linked documents
        # ----------------------------------------------------

        linked_documents_path = (
            output_path
            / "linked_documents.json"
        )

        linked_documents_path.write_text(
            json.dumps(
                linked_documents,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest_path = (
            output_path
            / "web_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 65)
        print(
            f"[SUCCESS] Saved "
            f"{len(documents)} web documents"
        )

        print(
            f"[DOCUMENT LINKS] "
            f"{len(linked_documents)}"
        )

        total_images = sum(
            item.get(
                "image_count",
                0,
            )
            for item in manifest
        )

        print(
            f"[IMAGE INFORMATION] "
            f"{total_images}"
        )

        print(
            f"[OUTPUT] "
            f"{output_path}"
        )

        print(
            f"[MANIFEST] "
            f"{manifest_path}"
        )

        print(
            f"[LINKED DOCS] "
            f"{linked_documents_path}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    BASE_URL = (
        "https://www.rajalakshmi.org/"
    )

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "web"
    )

    scraper = RECWebsiteScraper(
        base_url=BASE_URL,
        delay=0.5,
        timeout=25,
        max_pages=None,
        enable_ocr=True,
        max_ocr_images_per_page=30,
    )

    print("=" * 65)
    print("REC WEBSITE SCRAPER")
    print("=" * 65)

    print(
        f"Base URL: {BASE_URL}"
    )

    print(
        "Mode: COMPLETE REC WEBSITE CRAWL"
    )

    print(
        "Maximum pages: UNLIMITED"
    )

    print(
        f"OCR enabled: "
        f"{scraper.ocr_available}"
    )

    print("=" * 65)

    documents, linked_documents = (
        scraper.crawl(
            start_url=BASE_URL
        )
    )

    scraper.save_documents(
        documents,
        linked_documents,
        OUTPUT_DIR,
    )

    print()
    print("=" * 65)
    print("SCRAPING COMPLETED")
    print("=" * 65)

    print(
        f"Total website documents: "
        f"{len(documents)}"
    )

    print(
        f"Total linked documents: "
        f"{len(linked_documents)}"
    )