# src/ingest.py
"""Split every file in data/ into chunks, embed them, and store them in Qdrant.

Re-running always rebuilds the collection from scratch, so the stored chunks always match data/.
Each chunk keeps a stable ID made from its file name and position (md5), so the golden set's labels
("<file>#<position>") stay valid as long as the files do not change.

    python -m src.ingest
"""
import glob
import hashlib
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.embeddings import get_embedding
from src.core.vectorstore import count, recreate_collection, upsert_chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def load_chunks():
    """Every chunk of every file in data/, with its stable ID, source file and position."""
    chunks = []
    for path in sorted(glob.glob("data/*.txt")):
        filename = os.path.basename(path)
        with open(path) as f:
            text = f.read()
        for position, piece in enumerate(splitter.split_text(text)):
            chunk_id = hashlib.md5(f"{filename}-{position}".encode()).hexdigest()
            chunks.append({"id": chunk_id, "text": piece, "source": filename, "position": position})
    return chunks


def main():
    chunks = load_chunks()
    print(f"{len(chunks)} chunks from {len({chunk['source'] for chunk in chunks})} files")

    print("Embedding...")
    for done, chunk in enumerate(chunks, start=1):
        chunk["embedding"] = get_embedding(chunk["text"])
        if done % 20 == 0:
            print(f"  {done}/{len(chunks)}")

    # Rebuild only after every embedding succeeded: a failure above leaves the old collection untouched.
    recreate_collection()
    upsert_chunks(chunks)
    print(f"Done. Qdrant collection now has {count()} chunks.")


if __name__ == "__main__":
    main()
