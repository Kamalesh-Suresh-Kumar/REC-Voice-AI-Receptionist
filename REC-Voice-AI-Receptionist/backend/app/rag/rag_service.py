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
        top_k: int = 3,
    ):
        self.retriever = (
            retriever or Retriever()
        )

        self.llm = (
            llm or LLMService()
        )

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
            metadata = result.get(
                "metadata",
                {},
            )

            if not text:
                continue

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            context_parts.append(
                f"""
SOURCE {index}
File: {source}
Page: {page}

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

        # Retrieve relevant knowledge
        results = self.retriever.search(
            query=question,
            top_k=self.top_k,
        )

        # Build context
        context = self._build_context(
            results
        )

        # Generate answer
        answer = self.llm.generate(
            question=question,
            context=context,
        )

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            sources.append(
                {
                    "file": metadata.get(
                        "file"
                    ),
                    "page": metadata.get(
                        "page"
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
            metadata = result.get("metadata") or {}

            sources.append(
                {
                    "file": metadata.get("file"),
                    "page": metadata.get("page"),
                    "distance": result.get("distance"),
                    "query": result.get("search_query"),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved": results,
        }