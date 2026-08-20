from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb


class VectorStore:
    """
    Persistent ChromaDB vector store for the REC
    institutional knowledge base.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "rec_knowledge",
    ):
        # Project root:
        # REC-Voice-AI-Receptionist/
        #
        # This file is:
        # backend/app/rag/vector_store.py
        #
        # parents[3] = project root

        project_root = Path(__file__).resolve().parents[3]

        if persist_directory is None:
            persist_path = project_root / "vectorstore"
        else:
            persist_path = Path(persist_directory)

            if not persist_path.is_absolute():
                persist_path = project_root / persist_path

        persist_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[VECTORSTORE] Using: {persist_path}"
        )

        self.client = chromadb.PersistentClient(
            path=str(persist_path)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "REC institutional "
                        "knowledge base"
                    )
                },
            )
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """
        Add or update documents in ChromaDB.
        """

        # Chroma's runtime accepts normal Python
        # list[list[float]], but its type stubs are
        # stricter. cast() keeps Pylance happy.
        chroma_embeddings = cast(
            Any,
            embeddings,
        )
        
        chroma_metadatas = cast(
            Any,
            metadatas,
        )

        self.collection.upsert(
            documents=documents,
            embeddings=chroma_embeddings,
            metadatas=chroma_metadatas,
            ids=ids,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search the knowledge base using an embedding.
        """

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return cast(
            dict[str, Any],
            results,
        )

    def count(self) -> int:
        """
        Return number of indexed documents.
        """

        return self.collection.count()