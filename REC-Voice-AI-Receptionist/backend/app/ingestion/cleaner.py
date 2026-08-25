from __future__ import annotations

from pathlib import Path
import sys

# Allow running this file directly from backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.pdf_parser import extract_pdf
from app.ingestion.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


PROJECT_ROOT = BACKEND_DIR.parent

PDF_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pdf"
)

VECTORSTORE_DIRECTORY = (
    PROJECT_ROOT
    / "vectorstore"
)


def ingest_pdf(
    pdf_path: Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> int:

    print()
    print("=" * 60)
    print(f"Processing: {pdf_path.name}")
    print("=" * 60)

    pdf_document = extract_pdf(pdf_path)

    print(
        f"Pages with text: "
        f"{len(pdf_document.pages)}"
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    chunk_number = 0

    for page in pdf_document.pages:

        page_number = page["page"]
        page_text = page["text"]

        chunks = chunk_text(page_text)

        for chunk in chunks:

            documents.append(chunk)

            metadatas.append(
                {
                    "source": pdf_document.title,
                    "file": pdf_document.path,
                    "page": page_number,
                }
            )

            ids.append(
                f"{pdf_document.title}"
                f"_page_{page_number}"
                f"_chunk_{chunk_number}"
            )

            chunk_number += 1

    print(
        f"Created chunks: {len(documents)}"
    )

    if not documents:
        print(
            "WARNING: No text chunks were created."
        )
        return 0

    print()
    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        documents
    )

    print(
        f"Generated embeddings: "
        f"{len(embeddings)}"
    )

    print()
    print("Adding documents to ChromaDB...")

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"Indexed {len(documents)} chunks."
    )

    return len(documents)


def main() -> None:

    print("=" * 60)
    print("REC AI RECEPTIONIST - KNOWLEDGE INGESTION")
    print("=" * 60)

    print()
    print(f"PDF directory:")
    print(PDF_DIRECTORY)

    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: "
            f"{PDF_DIRECTORY}"
        )

    pdf_files = sorted(
        PDF_DIRECTORY.glob("*.pdf")
    )

    if not pdf_files:
        print()
        print(
            "No PDF files found."
        )
        print(
            f"Put PDFs inside: {PDF_DIRECTORY}"
        )
        return

    print()
    print(
        f"Found {len(pdf_files)} PDF(s)."
    )

    print()
    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        persist_directory=str(
            VECTORSTORE_DIRECTORY
        )
    )

    print()
    print(
        f"Existing vector count: "
        f"{vector_store.count()}"
    )

    total_chunks = 0

    for pdf_path in pdf_files:

        total_chunks += ingest_pdf(
            pdf_path=pdf_path,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"New chunks processed: "
        f"{total_chunks}"
    )

    print(
        f"Total ChromaDB documents: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":
    main()