from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
import hashlib
import json
import re
import time

import requests
from bs4 import BeautifulSoup


# ============================================================
# WEB DOCUMENT
# ============================================================

@dataclass
class WebDocument:
    url: str
    title: str
    text: str


# ============================================================
# REC WEBSITE SCRAPER
# ============================================================

class RECWebsiteScraper:
    def __init__(
        self,
        base_url: str,
        delay: float = 0.5,
        timeout: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "REC-AI-Receptionist-Research/1.0 "
                    "(educational project)"
                )
            }
        )

        # ----------------------------------------------------
        # Allowed domain
        # ----------------------------------------------------

        self.allowed_domain = urlparse(
            self.base_url
        ).netloc.lower()

        # ----------------------------------------------------
        # File types that should not be crawled
        # ----------------------------------------------------

        self.skip_extensions = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".ico",
            ".zip",
            ".rar",
            ".7z",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".mp3",
            ".wav",
            ".mp4",
            ".avi",
            ".mov",
            ".css",
            ".js",
            ".xml",
        }

        # ----------------------------------------------------
        # URLs that are not useful for the knowledge base
        # ----------------------------------------------------

        self.blocked_keywords = {
            "/login",
            "/signin",
            "/signup",
            "/register",
            "/logout",
            "/wp-admin",
        }
        
        # ----------------------------------------------------
        # Department pages
        # These pages are loaded through a dynamic menu on
        # the official website, so they are not discovered
        # by normal <a href=""> crawling.
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
        ]


    # ========================================================
    # URL NORMALIZATION
    # ========================================================

    def _normalize_url(
        self,
        url: str,
    ) -> str | None:

        if not url:
            return None

        url = url.strip()

        # Ignore non-web links
        if url.startswith(
            (
                "mailto:",
                "tel:",
                "javascript:",
                "#",
            )
        ):
            return None

        absolute_url = urljoin(
            self.base_url + "/",
            url,
        )

        parsed = urlparse(absolute_url)

        # Only HTTP/HTTPS
        if parsed.scheme not in {
            "http",
            "https",
        }:
            return None

        hostname = (
            parsed.hostname or ""
        ).lower()

        # Only official REC domain
        if hostname != self.allowed_domain:
            return None

        path = parsed.path or "/"

        # Remove duplicate slashes
        path = re.sub(
            r"/+",
            "/",
            path,
        )

        # Remove trailing slash except homepage
        if path != "/":
            path = path.rstrip("/")

        # Ignore unwanted file types
        extension = Path(path).suffix.lower()

        if extension in self.skip_extensions:
            return None

        # Ignore unwanted URL patterns
        lower_url = absolute_url.lower()

        for keyword in self.blocked_keywords:
            if keyword in lower_url:
                return None

        # Remove query parameters and fragments.
        # This prevents duplicate pages caused by tracking URLs.
        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                "",
                "",
                "",
            )
        )

        return normalized


    # ========================================================
    # SAME DOMAIN CHECK
    # ========================================================

    def _is_same_domain(
        self,
        url: str,
    ) -> bool:

        parsed_domain = (
            urlparse(url)
            .netloc
            .lower()
        )

        return parsed_domain == self.allowed_domain


    # ========================================================
    # CLEAN PAGE
    # ========================================================

    def _clean_page(
        self,
        soup: BeautifulSoup,
    ) -> str:

        # ----------------------------------------------------
        # Remove unnecessary HTML elements
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
            ]
        ):
            element.decompose()

        # ----------------------------------------------------
        # Prefer main/article content
        # ----------------------------------------------------

        main_content = soup.find("main")

        if main_content is None:
            main_content = soup.find("article")

        if main_content is None:
            main_content = soup.body

        if main_content is None:
            return ""

        # ----------------------------------------------------
        # Remove repeated website navigation
        # ----------------------------------------------------

        for element in main_content(
            [
                "header",
                "footer",
                "nav",
            ]
        ):
            element.decompose()

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        text = main_content.get_text(
            separator="\n"
        )

        # ----------------------------------------------------
        # Clean whitespace
        # ----------------------------------------------------

        lines: list[str] = []

        for line in text.splitlines():

            line = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(line)

        return "\n".join(lines)


    # ========================================================
    # FETCH PAGE
    # ========================================================

    def fetch(
        self,
        url: str,
    ) -> WebDocument | None:

        normalized_url = self._normalize_url(
            url
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
                f"[ERROR] {normalized_url}: {exc}"
            )

            return None

        # ----------------------------------------------------
        # Only process HTML
        # ----------------------------------------------------

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
        # Parse HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = ""

        if soup.title:
            title = soup.title.get_text(
                strip=True
            )

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        text = self._clean_page(
            soup
        )

        if not text:
            return None

        return WebDocument(
            url=normalized_url,
            title=title,
            text=text,
        )


    # ========================================================
    # DISCOVER LINKS
    # ========================================================

    def discover_links(
        self,
        start_url: str,
    ) -> list[str]:

        try:

            response = self.session.get(
                start_url,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            print(
                f"[ERROR] Link discovery failed: "
                f"{start_url}"
            )

            return []

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        links: set[str] = set()

        for anchor in soup.find_all(
            "a",
            href=True,
        ):

            href = anchor.get("href")

            if not isinstance(
                href,
                str,
            ):
                continue

            normalized_url = self._normalize_url(
                urljoin(
                    start_url,
                    href,
                )
            )

            if normalized_url is None:
                continue

            if not self._is_same_domain(
                normalized_url
            ):
                continue

            links.add(
                normalized_url
            )

        return sorted(links)


    # ========================================================
    # CRAWL WEBSITE
    # ========================================================

    def crawl(
        self,
        start_url: str | None = None,
        max_pages: int = 50,
    ) -> list[WebDocument]:

        if start_url is None:
            start_url = self.base_url

        normalized_start = self._normalize_url(
            start_url
        )

        if normalized_start is None:
            raise ValueError(
                "Invalid starting URL"
            )

        visited: set[str] = set()

        queue: list[str] = [
            normalized_start
        ]

        # Add department pages explicitly because the
        # department menu is dynamically generated.
        for department_path in self.department_urls:

            department_url = self._normalize_url(
                department_path
            )

            if (
                department_url is not None
                and department_url not in queue
            ):
                queue.append(
                    department_url
                )

        documents: list[WebDocument] = []

        while (
            queue
            and len(documents) < max_pages
        ):

            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            print(
                f"[{len(documents) + 1}/{max_pages}] "
                f"Fetching: {url}"
            )

            document = self.fetch(
                url
            )

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

                # Ignore pages with almost no useful content
                if word_count >= 20:

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
                # Discover more pages
                # ------------------------------------------------

                try:

                    links = self.discover_links(
                        url
                    )

                    new_links = 0

                    for link in links:

                        if (
                            link not in visited
                            and link not in queue
                        ):

                            queue.append(
                                link
                            )

                            new_links += 1

                    print(
                        f"    New links: "
                        f"{new_links}"
                    )

                except Exception as exc:

                    print(
                        f"    [WARNING] "
                        f"Link discovery failed: "
                        f"{exc}"
                    )

            time.sleep(
                self.delay
            )

        return documents


    # ========================================================
    # SAVE SCRAPED DOCUMENTS
    # ========================================================

    def save_documents(
        self,
        documents: list[WebDocument],
        output_dir: str | Path,
    ) -> None:

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest: list[dict] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            # Create stable filename
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
                output_path / filename
            )

            content = (
                f"TITLE: "
                f"{document.title}\n"
                f"SOURCE_URL: "
                f"{document.url}\n"
                f"SOURCE_TYPE: website\n"
                f"\n"
                f"{document.text}\n"
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
                    "word_count": len(
                        document.text.split()
                    ),
                }
            )

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
        print(
            f"[SUCCESS] Saved "
            f"{len(documents)} web documents"
        )

        print(
            f"[OUTPUT] "
            f"{output_path}"
        )

        print(
            f"[MANIFEST] "
            f"{manifest_path}"
        )


# ============================================================
# TEST / DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    # Official REC website
    BASE_URL = (
        "https://www.rajalakshmi.org/"
    )

    # Project root:
    # backend/app/ingestion/web_scraper.py
    #                  ↑
    # parents[3] = project root

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
        timeout=20,
    )

    print("=" * 65)
    print("REC WEBSITE SCRAPER")
    print("=" * 65)

    documents = scraper.crawl(
        start_url=BASE_URL,
        max_pages=50,
    )

    scraper.save_documents(
        documents,
        OUTPUT_DIR,
    )

    print()
    print("=" * 65)
    print("SCRAPING COMPLETED")
    print("=" * 65)

    print(
        f"Total documents: "
        f"{len(documents)}"
    )