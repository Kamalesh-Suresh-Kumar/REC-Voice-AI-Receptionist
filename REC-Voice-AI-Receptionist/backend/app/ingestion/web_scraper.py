from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time

import requests
from bs4 import BeautifulSoup


@dataclass
class WebDocument:
    url: str
    title: str
    text: str


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

    def _is_same_domain(self, url: str) -> bool:
        base_domain = urlparse(self.base_url).netloc
        current_domain = urlparse(url).netloc

        return current_domain == base_domain

    def _clean_page(self, soup: BeautifulSoup) -> str:
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
            ]
        ):
            element.decompose()

        text = soup.get_text(separator="\n")

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)

    def fetch(self, url: str) -> WebDocument | None:
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            print(f"[ERROR] {url}: {exc}")
            return None

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        title = ""

        if soup.title:
            title = soup.title.get_text(strip=True)

        text = self._clean_page(soup)

        return WebDocument(
            url=url,
            title=title,
            text=text,
        )

    def discover_links(self, start_url: str) -> list[str]:
        response = self.session.get(
            start_url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        links: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            absolute_url = urljoin(
                start_url,
                href,
            )

            absolute_url = absolute_url.split("#")[0]

            if self._is_same_domain(absolute_url):
                links.add(absolute_url)

        return sorted(links)

    def crawl(
        self,
        start_url: str,
        max_pages: int = 30,
    ) -> list[WebDocument]:

        visited: set[str] = set()
        queue: list[str] = [start_url]

        documents: list[WebDocument] = []

        while queue and len(documents) < max_pages:
            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            print(
                f"[{len(documents) + 1}/{max_pages}] "
                f"Fetching: {url}"
            )

            document = self.fetch(url)

            if document and document.text:
                documents.append(document)

                try:
                    links = self.discover_links(url)

                    for link in links:
                        if link not in visited:
                            queue.append(link)

                except requests.RequestException:
                    pass

            time.sleep(self.delay)

        return documents