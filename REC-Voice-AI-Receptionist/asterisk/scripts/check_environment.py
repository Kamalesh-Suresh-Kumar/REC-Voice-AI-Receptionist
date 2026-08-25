import sys


def check_package(name, import_name=None):
    import_name = import_name or name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")


print("=" * 60)
print("REC AI RECEPTIONIST - ENVIRONMENT CHECK")
print("=" * 60)

print(f"\nPython: {sys.version}")

packages = [
    ("fastapi", "fastapi"),
    ("chromadb", "chromadb"),
    ("sentence-transformers", "sentence_transformers"),
    ("pymupdf", "pymupdf"),
    ("beautifulsoup4", "bs4"),
    ("faster-whisper", "faster_whisper"),
]

print()

for package, import_name in packages:
    check_package(package, import_name)