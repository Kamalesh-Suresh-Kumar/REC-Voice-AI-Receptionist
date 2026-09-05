from app.rag.retriever import Retriever


print("=" * 60)
print("REC MULTI-TOPIC RETRIEVER TEST")
print("=" * 60)

retriever = Retriever()

queries = [
    "Discrete Mathematical Structures objectives course structure",
    "Problem Solving and Python Programming objectives course structure",
]

results = retriever.search_multiple(
    queries,
    top_k=3,
)

print()
print(f"Retrieved {len(results)} results.")
print()

for index, result in enumerate(results, start=1):

    print("-" * 60)
    print(f"RESULT {index}")
    print("-" * 60)

    print(
        "Search query:",
        result["search_query"],
    )

    print(
        "Distance:",
        result["distance"],
    )

    print(
        "Metadata:",
        result["metadata"],
    )

    print(
        "Text:",
        result["text"][:1000],
    )

    print()