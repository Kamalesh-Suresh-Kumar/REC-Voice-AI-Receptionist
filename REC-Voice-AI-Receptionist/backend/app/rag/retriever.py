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

        # Convert query into the same embedding space
        # used when documents were indexed.
        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
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

            retrieved.append(
                {
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        return retrieved
    
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