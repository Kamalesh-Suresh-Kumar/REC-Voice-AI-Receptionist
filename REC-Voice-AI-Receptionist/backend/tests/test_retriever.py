from app.rag.retriever import Retriever


def main():

    print("=" * 60)
    print("REC RETRIEVER TEST")
    print("=" * 60)

    retriever = Retriever(
        top_k=5
    )

    question = input(
        "\nQuestion: "
    ).strip()

    results = retriever.search(
        question
    )

    print(
        f"\nRetrieved {len(results)} results."
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

        print(
            "Distance:",
            result.get("distance")
        )

        print(
            "Metadata:",
            result.get("metadata")
        )

        print(
            "Text:"
        )

        print(
            result.get("text", "")[:1000]
        )


if __name__ == "__main__":
    main()