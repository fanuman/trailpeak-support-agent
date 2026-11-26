"""List every chunk in the vector store with its position and section, so golden-set questions can be
labelled with the sections that answer them.

Usage (Qdrant running, as for the evaluation harness):
    python -m src.evaluation.inspect_chunks                      # one line per chunk: position, size, section
    python -m src.evaluation.inspect_chunks returns_policy.txt   # full text of one file's chunks
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from src.core.vectorstore import all_chunks


def chunk_label(chunk):
    """A chunk's position name: '<file>#<position>', e.g. 'returns_policy.txt#0'.
    Changes whenever chunking changes, so it is for reports, not for golden-set labels."""
    return f"{chunk['source']}#{chunk['position']}"


def section_label(chunk):
    """The golden set's name for where a chunk's content lives: '<file> > <section>',
    e.g. 'returns_policy.txt > REFUND TIMING'. Stays the same when chunking changes."""
    return f"{chunk['source']} > {chunk['section']}"


def main(only_file=None):
    chunks = sorted((c for c in all_chunks() if not only_file or c["source"] == only_file),
                    key=lambda c: (c["source"], c["position"]))
    for chunk in chunks:
        if only_file:
            print(f"===== {chunk_label(chunk)} | {section_label(chunk)} =====\n{chunk['text']}\n")
        else:
            print(f"{chunk_label(chunk):<26} {len(chunk['text']):>5}  {section_label(chunk)}")
    print(f"\n{len(chunks)} chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
