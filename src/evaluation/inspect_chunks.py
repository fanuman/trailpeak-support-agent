"""List every chunk in the vector store with a readable label, so golden-set questions can be
labelled with the exact chunks that answer them.

Usage (Qdrant running, as for the evaluation harness):
    python -m src.evaluation.inspect_chunks                      # one line per chunk
    python -m src.evaluation.inspect_chunks returns_policy.txt   # full text of one file's chunks
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from src.core.vectorstore import all_chunks


def chunk_label(chunk):
    """The golden set's name for a chunk: '<file>#<position>', e.g. 'returns_policy.txt#0'."""
    return f"{chunk['source']}#{chunk['position']}"


def main(only_file=None):
    chunks = sorted((c for c in all_chunks() if not only_file or c["source"] == only_file),
                    key=lambda c: (c["source"], c["position"]))
    for chunk in chunks:
        if only_file:
            print(f"===== {chunk_label(chunk)} =====\n{chunk['text']}\n")
        else:
            print(f"{chunk_label(chunk):<26} {' '.join(chunk['text'].split())[:80]}")
    print(f"\n{len(chunks)} chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
