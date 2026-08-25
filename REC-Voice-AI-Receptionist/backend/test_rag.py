from app.rag.rag_service import RAGService


def main():

    print("=" * 60)
    print("REC AI RECEPTIONIST - RAG TEST")
    print("=" * 60)

    rag = RAGService(
        top_k=5
    )

    question = input(
        "\nAsk REC a question: "
    ).strip()

    if not question:
        print("No question entered.")
        return

    print("\n[1] Retrieving knowledge...")

    result = rag.answer(
        question
    )

    print("\n[2] ANSWER")
    print("-" * 60)
    print(result["answer"])

    print("\n[3] SOURCES")
    print("-" * 60)

    for source in result["sources"]:
        print(source)

    print("\n[4] RETRIEVED CHUNKS")
    print("-" * 60)

    for index, item in enumerate(
        result["results"],
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            "Distance:",
            item.get("distance")
        )

        print(
            item.get("text", "")[:500]
        )


if __name__ == "__main__":
    main()