"""List every chunk in the Chroma collection with a readable label, so
golden-set questions can be labelled with the exact chunks that answer them.

Usage (Chroma running, as for the evaluation harness):
    python -m src.evaluation.inspect_chunks                      # one line per chunk
    python -m src.evaluation.inspect_chunks returns_policy.txt   # full text of one file's chunks
"""
import hashlib
import sys

from dotenv import load_dotenv

load_dotenv()

from src.core.vectorstore import get_collection

MAX_CHUNKS_PER_FILE = 1000


def chunk_position(source, chunk_id):
    """Recover a chunk's position in its file from its ID.

    ingest.py builds each ID as md5("<file>-<position>"), so trying positions
    0, 1, 2, ... until the hash matches gives the position back.
    """
    for position in range(MAX_CHUNKS_PER_FILE):
        if hashlib.md5(f"{source}-{position}".encode()).hexdigest() == chunk_id:
            return position
    return -1


def main(only_file=None):
    data = get_collection().get(include=["documents", "metadatas"])
    rows = []
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        source = meta["source"]
        if only_file and source != only_file:
            continue
        rows.append((source, chunk_position(source, chunk_id), text))

    for source, position, text in sorted(rows):
        label = f"{source}#{position}"
        if only_file:
            print(f"===== {label} =====\n{text}\n")
        else:
            print(f"{label:<26} {' '.join(text.split())[:80]}")
    print(f"\n{len(rows)} chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
