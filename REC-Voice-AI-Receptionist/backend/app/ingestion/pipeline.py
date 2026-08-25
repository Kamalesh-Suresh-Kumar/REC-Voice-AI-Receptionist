from __future__ import annotations

import hashlib
from pathlib import Path

from app.ingestion.pdf_parser import extract_pdf
from app.ingestion.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


# Project root:
# REC-Voice-AI-Receptionist/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Your PDF location:
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pdf"


def make_id(
    source: str,
    page: int,
    chunk_index: int,
) -> str:
    """
    Create a stable unique ID for every chunk.
    """

    value = (
        f"{source}|page={page}|chunk={chunk_index}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def ingest_pdf(
    pdf_path: Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> int:

    print("=" * 60)
    print("REC AI RECEPTIONIST - PDF INGESTION")
    print("=" * 60)

    print(f"[PDF] {pdf_path}")

    # --------------------------------------------------
    # 1. Extract PDF
    # --------------------------------------------------

    document = extract_pdf(pdf_path)

    print(
        f"[PDF] Title: {document.title}"
    )

    print(
        f"[PDF] Pages with text: "
        f"{len(document.pages)}"
    )

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for page_data in document.pages:

        page_number = int(
            page_data["page"]
        )

        page_text = str(
            page_data["text"]
        )

        chunks = chunk_text(
            page_text,
            chunk_size=1200,
            overlap=400,
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            chunk = chunk.strip()

            if not chunk:
                continue

            documents.append(chunk)

            metadatas.append(
                {
                    "source": document.title,
                    "file": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_index,
                }
            )

            ids.append(
                make_id(
                    document.title,
                    page_number,
                    chunk_index,
                )
            )

    print(
        f"[CHUNKS] Created: {len(documents)}"
    )

    if not documents:
        print(
            "[ERROR] No text chunks were created."
        )
        return 0

    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    print(
        "[EMBEDDINGS] Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        documents
    )

    print(
        f"[EMBEDDINGS] Created: "
        f"{len(embeddings)}"
    )

    # --------------------------------------------------
    # 4. Store in ChromaDB
    # --------------------------------------------------

    print(
        "[VECTORSTORE] Storing documents..."
    )

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"[VECTORSTORE] Collection count: "
        f"{vector_store.count()}"
    )

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    return len(documents)


def main() -> None:

    print(
        f"[PROJECT] {PROJECT_ROOT}"
    )

    print(
        f"[PDF DIRECTORY] {PDF_DIRECTORY}"
    )

    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory not found: "
            f"{PDF_DIRECTORY}"
        )

    pdf_files = sorted(
        PDF_DIRECTORY.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: "
            f"{PDF_DIRECTORY}"
        )

    print(
        f"[PDF FILES] Found: "
        f"{len(pdf_files)}"
    )

    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Load embedding model once.
    embedding_model = EmbeddingModel()

    # Connect to ChromaDB.
    # Since you run this from backend/,
    # this will use backend/vectorstore/.
    vector_store = VectorStore()

    print()
    print("[VECTORSTORE] Existing chunks:", vector_store.count())

    vector_store.reset()

    print("[VECTORSTORE] After reset:", vector_store.count())

    total_chunks = 0

    for pdf_path in pdf_files:

        total_chunks += ingest_pdf(
            pdf_path=pdf_path,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    print()
    print("=" * 60)
    print("FINAL INGESTION RESULT")
    print("=" * 60)
    print(
        f"PDFs processed: {len(pdf_files)}"
    )
    print(
        f"Chunks added: {total_chunks}"
    )
    print(
        f"ChromaDB count: {vector_store.count()}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()