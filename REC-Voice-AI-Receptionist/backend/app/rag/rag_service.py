from __future__ import annotations

from typing import Any

from .retriever import Retriever
from .llm_service import LLMService


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Flow:

        Question
           ↓
        Retriever
           ↓
        Relevant documents
           ↓
        LLM
           ↓
        Answer
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMService | None = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMService()

        self.top_k = top_k

    def _build_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:

        context_parts: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            text = result.get("text", "")
            metadata = result.get("metadata") or {}

            if not text:
                continue

            source = metadata.get(
                "source",
                metadata.get("file", "Unknown"),
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            section = metadata.get(
                "section",
                "",
            )

            section_text = ""

            if section:
                section_text = f"\nSection: {section}"

            context_parts.append(
                f"""
SOURCE {index}
File: {source}
Page: {page}{section_text}

{text}
"""
            )

        return "\n".join(context_parts)

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:

        if not question.strip():
            return {
                "answer": (
                    "I'm sorry, I didn't "
                    "catch the question."
                ),
                "sources": [],
                "retrieved": [],
            }

        # --------------------------------------------------
        # RETRIEVE
        # --------------------------------------------------

        results = self.retriever.search(
            query=question,
            top_k=self.top_k,
        )
        
        # ==========================================================
        # PERSONAL / PRIVATE QUERY GUARD
        # ==========================================================

        q_lower = question.lower().strip()

        personal_query = (
            "my personal" in q_lower
            or "my current" in q_lower
            or "my individual" in q_lower
            or "my attendance" in q_lower
            or "my outstanding fee" in q_lower
            or "my fee amount" in q_lower
            or "my examination results" in q_lower
            or "my exam results" in q_lower
            or "individual semester examination results" in q_lower
            or (
                "salary" in q_lower
                and (
                    "faculty" in q_lower
                    or "professor" in q_lower
                    or "staff" in q_lower
                )
            )
        )

        if personal_query:
            return {
                "answer": (
                    "I'm sorry, I don't have enough information "
                    "to answer that. Please contact reception."
                ),
                "retrieved": [],
            }

        # --------------------------------------------------
        # BUILD CONTEXT
        # --------------------------------------------------

        context = self._build_context(results)

        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        # --------------------------------------------------
        # SOURCE INFORMATION
        # --------------------------------------------------

        sources = []

        for result in results:

            metadata = result.get(
                "metadata"
            ) or {}

            sources.append(
                {
                    "file": metadata.get(
                        "file",
                        metadata.get("source"),
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                    "section": metadata.get(
                        "section"
                    ),
                    "distance": result.get(
                        "distance"
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved": results,
        }

    def ask_comparison(
        self,
        question: str,
        topics: list[str],
    ) -> dict[str, Any]:

        queries = [
            f"{topic} objectives course structure"
            for topic in topics
        ]

        results = self.retriever.search_multiple(
            queries=queries,
            top_k=self.top_k,
        )

        context = self._build_context(results)

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        sources = []

        for result in results:

            metadata = result.get(
                "metadata"
            ) or {}

            sources.append(
                {
                    "file": metadata.get(
                        "file",
                        metadata.get("source"),
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                    "section": metadata.get(
                        "section"
                    ),
                    "distance": result.get(
                        "distance"
                    ),
                    "query": result.get(
                        "search_query"
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved": results,
        }