from __future__ import annotations

import hashlib
from pathlib import Path

from app.ingestion.pdf_parser import extract_pdf
from app.ingestion.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# REC-Voice-AI-Receptionist/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# PDF data
PDF_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pdf"
)

# Website data
WEB_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "web"
)


# ============================================================
# CHUNK ID
# ============================================================

def make_id(
    source: str,
    location: str,
    chunk_index: int,
) -> str:
    """
    Create a stable unique ID for every chunk.
    """

    value = (
        f"{source}|"
        f"location={location}|"
        f"chunk={chunk_index}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    pdf_path: Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> int:

    print("=" * 60)
    print("REC AI RECEPTIONIST - PDF INGESTION")
    print("=" * 60)

    print(f"[PDF] {pdf_path}")

    # --------------------------------------------------------
    # 1. Extract PDF
    # --------------------------------------------------------

    document = extract_pdf(
        pdf_path
    )

    print(
        f"[PDF] Title: "
        f"{document.title}"
    )

    print(
        f"[PDF] Pages with text: "
        f"{len(document.pages)}"
    )

    # --------------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------------

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

            documents.append(
                chunk
            )

            metadatas.append(
                {
                    "source": document.title,
                    "file": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_index,
                    "source_type": "pdf",
                }
            )

            ids.append(
                make_id(
                    document.title,
                    f"page={page_number}",
                    chunk_index,
                )
            )

    print(
        f"[CHUNKS] Created: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "[ERROR] No text chunks "
            "were created."
        )

        return 0

    # --------------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. Store in ChromaDB
    # --------------------------------------------------------

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
    print("PDF INGESTION COMPLETE")
    print("=" * 60)

    return len(documents)


# ============================================================
# WEBSITE INGESTION
# ============================================================

def ingest_web_document(
    web_path: Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> int:

    print("=" * 60)
    print("REC AI RECEPTIONIST - WEBSITE INGESTION")
    print("=" * 60)

    print(
        f"[WEB] File: {web_path.name}"
    )

    # --------------------------------------------------------
    # 1. Read scraped website document
    # --------------------------------------------------------

    try:

        content = web_path.read_text(
            encoding="utf-8"
        )

    except OSError as exc:

        print(
            f"[ERROR] Could not read "
            f"{web_path}: {exc}"
        )

        return 0

    if not content.strip():

        print(
            "[ERROR] Website file is empty."
        )

        return 0

    # --------------------------------------------------------
    # 2. Extract metadata
    # --------------------------------------------------------

    title = web_path.stem

    source_url = ""

    lines = content.splitlines()

    for line in lines:

        if line.startswith("TITLE:"):

            title = line[
                len("TITLE:"):
            ].strip()

        elif line.startswith(
            "SOURCE_URL:"
        ):

            source_url = line[
                len("SOURCE_URL:"):
            ].strip()

    # --------------------------------------------------------
    # 3. Remove metadata header before chunking
    # --------------------------------------------------------

    text_lines: list[str] = []

    for line in lines:

        if line.startswith(
            "TITLE:"
        ):
            continue

        if line.startswith(
            "SOURCE_URL:"
        ):
            continue

        if line.startswith(
            "SOURCE_TYPE:"
        ):
            continue

        text_lines.append(
            line
        )

    text = "\n".join(
        text_lines
    ).strip()

    if not text:

        print(
            "[ERROR] No website text "
            "available after cleaning."
        )

        return 0

    # --------------------------------------------------------
    # 4. Create chunks
    # --------------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=1200,
        overlap=400,
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for chunk_index, chunk in enumerate(
        chunks
    ):

        chunk = chunk.strip()

        if not chunk:
            continue

        documents.append(
            chunk
        )

        metadatas.append(
            {
                "source": title,
                "file": web_path.name,
                "url": source_url,
                "chunk": chunk_index,
                "source_type": "website",
            }
        )

        ids.append(
            make_id(
                web_path.name,
                source_url or title,
                chunk_index,
            )
        )

    print(
        f"[WEB] Title: {title}"
    )

    print(
        f"[WEB] URL: {source_url}"
    )

    print(
        f"[CHUNKS] Created: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "[ERROR] No website chunks "
            "were created."
        )

        return 0

    # --------------------------------------------------------
    # 5. Create embeddings
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Store in ChromaDB
    # --------------------------------------------------------

    print(
        "[VECTORSTORE] Storing "
        "website documents..."
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
    print("WEBSITE INGESTION COMPLETE")
    print("=" * 60)

    return len(documents)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("REC AI RECEPTIONIST - FULL INGESTION")
    print("=" * 60)

    print(
        f"[PROJECT] {PROJECT_ROOT}"
    )

    print(
        f"[PDF DIRECTORY] "
        f"{PDF_DIRECTORY}"
    )

    print(
        f"[WEB DIRECTORY] "
        f"{WEB_DIRECTORY}"
    )

    # ========================================================
    # CHECK PDF DIRECTORY
    # ========================================================

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

    print()
    print(
        f"[PDF FILES] Found: "
        f"{len(pdf_files)}"
    )

    for pdf in pdf_files:

        print(
            f"  - {pdf.name}"
        )

    # ========================================================
    # CHECK WEBSITE DIRECTORY
    # ========================================================

    if not WEB_DIRECTORY.exists():

        raise FileNotFoundError(
            f"Website directory not found: "
            f"{WEB_DIRECTORY}"
        )

    web_files = sorted(
        WEB_DIRECTORY.glob("*.txt")
    )

    if not web_files:

        raise FileNotFoundError(
            f"No website text files "
            f"found in: {WEB_DIRECTORY}"
        )

    print()
    print(
        f"[WEB FILES] Found: "
        f"{len(web_files)}"
    )

    for web_file in web_files:

        print(
            f"  - {web_file.name}"
        )

    # ========================================================
    # LOAD EMBEDDING MODEL ONCE
    # ========================================================

    print()
    print(
        "[EMBEDDINGS] Loading model..."
    )

    embedding_model = EmbeddingModel()

    # ========================================================
    # CONNECT TO CHROMADB
    # ========================================================

    vector_store = VectorStore()

    print()
    print(
        "[VECTORSTORE] Existing chunks:",
        vector_store.count(),
    )

    # ========================================================
    # IMPORTANT:
    # REBUILD DATABASE FROM PDF + WEBSITE
    # ========================================================

    print()
    print(
        "[VECTORSTORE] Resetting "
        "existing collection..."
    )

    vector_store.reset()

    print(
        "[VECTORSTORE] After reset:",
        vector_store.count(),
    )

    # ========================================================
    # INGEST PDFs
    # ========================================================

    total_pdf_chunks = 0

    for pdf_path in pdf_files:

        total_pdf_chunks += ingest_pdf(
            pdf_path=pdf_path,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    # ========================================================
    # INGEST WEBSITE
    # ========================================================

    total_web_chunks = 0

    for web_path in web_files:

        total_web_chunks += ingest_web_document(
            web_path=web_path,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL INGESTION RESULT")
    print("=" * 60)

    print(
        f"PDFs processed: "
        f"{len(pdf_files)}"
    )

    print(
        f"PDF chunks added: "
        f"{total_pdf_chunks}"
    )

    print(
        f"Website pages processed: "
        f"{len(web_files)}"
    )

    print(
        f"Website chunks added: "
        f"{total_web_chunks}"
    )

    print(
        f"Total chunks added: "
        f"{total_pdf_chunks + total_web_chunks}"
    )

    print(
        f"ChromaDB count: "
        f"{vector_store.count()}"
    )

    print("=" * 60)
    print("FULL INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()