# src/ingest.py
"""Split every file in data/ into chunks, embed them, and store them in Qdrant.

Chunks follow the documents' own structure: each file is split into sections at its headings
(a product in the catalog, a topic in a policy), and a section is only split further if it is
longer than the splitter's chunk_size. Every chunk starts with a header naming its document,
section and, for products, the SKU, so a chunk always says what it is about.

Re-running always rebuilds the collection from scratch, so the stored chunks always match data/.
Each chunk keeps a stable ID made from its file name and position (md5). The golden set labels
sections ("<file> > <section>"), not positions, so its labels survive chunking changes.

    python -m src.ingest
"""
import glob
import hashlib
import os
import re

from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.embeddings import get_embedding
from src.core.vectorstore import count, recreate_collection, upsert_chunks

# Sections are 200-1,250 characters, so each becomes one chunk; longer future sections are split
# inside the section only (overlap never crosses into another section).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def is_heading(line):
    """A section title: capital letters only (text in brackets may be mixed case), no colon,
    not a list item. 'REFUND TIMING', 'DOMESTIC SHIPPING (United States)' and
    'HYDROFLOW WATER FILTER' are titles; 'SKU: TP-FILT-007' and '- Final Sale items' are not."""
    stripped = line.strip()
    if stripped.startswith("-") or ":" in stripped:
        return False
    outside_brackets = re.sub(r"\([^)]*\)", "", stripped)
    letters = [c for c in outside_brackets if c.isalpha()]
    return len(letters) >= 3 and not any(c.islower() for c in outside_brackets)


def split_sections(text, default_title):
    """Return (document title, [(section title, section text), ...]).

    The first heading is the document's title; any text right under it forms a section named
    after the document (about_us.txt is all one such section). The catalog's ===== rules are dropped.
    """
    title, current, lines, sections = None, default_title, [], []
    for line in text.splitlines():
        if line.strip() and set(line.strip()) == {"="}:
            continue
        if is_heading(line):
            if title is None:
                title = current = line.strip()
                continue
            sections.append((current, "\n".join(lines).strip()))
            current, lines = line.strip(), []
        else:
            lines.append(line)
    sections.append((current, "\n".join(lines).strip()))
    return title or default_title, [(name, body) for name, body in sections if body]


def chunk_header(title, section, body):
    """'<document> | <section> | SKU <code>': what every chunk of this section is about."""
    sku = re.search(r"^SKU:\s*(\S+)", body, re.MULTILINE)
    parts = [title] + ([section] if section != title else []) + ([f"SKU {sku.group(1)}"] if sku else [])
    return " | ".join(parts)


def load_chunks():
    """Every chunk of every file in data/, with its stable ID, source file, section and position."""
    chunks = []
    for path in sorted(glob.glob("data/*.txt")):
        filename = os.path.basename(path)
        with open(path) as f:
            title, sections = split_sections(f.read(), default_title=filename)
        position = 0
        for section, body in sections:
            header = chunk_header(title, section, body)
            for piece in splitter.split_text(body):
                chunk_id = hashlib.md5(f"{filename}-{position}".encode()).hexdigest()
                chunks.append({"id": chunk_id, "text": f"{header}\n\n{piece}", "source": filename,
                               "section": section, "position": position})
                position += 1
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
