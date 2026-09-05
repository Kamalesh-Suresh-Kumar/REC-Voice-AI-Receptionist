from __future__ import annotations

from typing import Any

from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant REC knowledge-base
    documents for a user query.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = 5,
    ):
        self.embedding_model = (
            embedding_model or EmbeddingModel()
        )

        self.vector_store = (
            vector_store or VectorStore()
        )

        self.top_k = top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            return []

        k = top_k or self.top_k

        # --------------------------------------------------
        # 1. Semantic vector search
        # --------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        # Retrieve more candidates than we finally return.
        candidate_k = max(k * 10, 50)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        documents_raw = results.get("documents")
        metadatas_raw = results.get("metadatas")
        distances_raw = results.get("distances")

        documents = (
            documents_raw[0]
            if documents_raw
            else []
        )

        metadatas = (
            metadatas_raw[0]
            if metadatas_raw
            else []
        )

        distances = (
            distances_raw[0]
            if distances_raw
            else []
        )

        retrieved: list[dict[str, Any]] = []

        # Normalize query for exact phrase matching.
        query_normalized = " ".join(
            query.lower().split()
        )

        # Useful terms for exact matching.
        query_terms = [
            term.strip(".,?!:;()[]{}\"'")
            for term in query_normalized.split()
            if len(term.strip(".,?!:;()[]{}\"'")) >= 3
        ]

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            text = document or ""
            text_normalized = " ".join(
                text.lower().split()
            )

            # --------------------------------------------------
            # 2. Exact-match scoring
            # --------------------------------------------------

            exact_score = 0.0

            # Exact full-query phrase.
            if query_normalized in text_normalized:
                exact_score += 10.0

            # Count matching query terms.
            if query_terms:
                matching_terms = sum(
                    1
                    for term in query_terms
                    if term in text_normalized
                )

                exact_score += (
                    matching_terms / len(query_terms)
                ) * 3.0

            # --------------------------------------------------
            # Important factual-term boosting
            # --------------------------------------------------

            factual_keywords = [
                "credits",
                "prescribed",
                "degree",
                "programme",
                "subject",
                "code",
            ]

            for keyword in factual_keywords:
                if keyword in query_normalized and keyword in text_normalized:
                    exact_score += 2.0

            # Strong boost when a course code from the
            # question appears directly in the document.
            import re

            course_codes = re.findall(
                r"\b[A-Z]{2,5}\d{4,6}\b",
                query.upper(),
            )

            for course_code in course_codes:
                if course_code.lower() in text_normalized:
                    exact_score += 20.0

            # --------------------------------------------------
            # 3. Course-name detection
            # --------------------------------------------------

            course_keywords = [
                "theory of computation",
                "discrete mathematical structures",
                "problem solving and python programming",
                "digital logic and microprocessor",
                "data structures",
                "operating systems",
                "computer networks",
                "database management systems",
            ]

            for course_name in course_keywords:

                if course_name in query_normalized:

                    if course_name in text_normalized:
                        exact_score += 15.0

            retrieved.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                    "exact_score": exact_score,
                }
            )

        # --------------------------------------------------
        # 4. Combined ranking
        # --------------------------------------------------

        def ranking_score(
            item: dict[str, Any],
        ) -> float:

            distance = item.get("distance")

            if distance is None:
                distance_score = 0.0
            else:
                # Smaller Chroma distance = better.
                distance_score = -float(distance)

            exact_score = float(
                item.get("exact_score", 0.0)
            )

            return (
                exact_score
                + distance_score
            )

        retrieved.sort(key=ranking_score, reverse=True)
        
        # --------------------------------------------------
        # 5. Return final results
        # --------------------------------------------------

        return retrieved[:k]

    def search_multiple(
    self,
    queries: list[str],
    top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search independently for multiple topics
        and combine the results.
        """

        all_results: list[dict[str, Any]] = []
        seen: set[tuple] = set()

        for query in queries:

            if not query.strip():
                continue

            results = self.search(
            query=query,
            top_k=top_k,
            )

            for result in results:

                metadata = result.get("metadata") or {}

                key = (
                metadata.get("file"),
                metadata.get("page"),
                metadata.get("chunk"),
                )

                if key in seen:
                    continue

                seen.add(key)

                result["search_query"] = query

                all_results.append(result)

        return all_results