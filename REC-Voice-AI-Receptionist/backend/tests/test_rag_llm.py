from app.rag.rag_service import RAGService


print("=" * 60)
print("REC AI RECEPTIONIST - COMPARISON RAG TEST")
print("=" * 60)

rag = RAGService()

question = (
    "Compare Discrete Mathematical Structures "
    "with Problem Solving and Python Programming "
    "in terms of objectives and course structure."
)

topics = [
    "Discrete Mathematical Structures",
    "Problem Solving and Python Programming",
]

print()
print("QUESTION:")
print(question)

print()
print("[1] RETRIEVING KNOWLEDGE...")

result = rag.ask_comparison(
    question=question,
    topics=topics,
)

print()
print("[2] ANSWER")
print("-" * 60)
print(result["answer"])

print()
print("[3] SOURCES")
print("-" * 60)

for source in result["sources"]:
    print(source)