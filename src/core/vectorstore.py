"""The vector store: Qdrant, holding every chunk with two vectors.

Each chunk is one Qdrant point:
  id       a UUID made from the chunk's md5 ID (Qdrant ids must be UUIDs or integers)
  vectors  "dense": the OpenAI embedding (text-embedding-3-small, 1536 numbers) - search by meaning
           "bm25":  a sparse BM25 vector made locally by FastEmbed - search by exact words
  payload  {"text", "source", "position"}  (position = the chunk's number within its file)
Qdrant applies IDF to "bm25" at search time (modifier=IDF), from the current collection.

Only this file talks to the Qdrant client. The rest of the app gets plain dicts back, so it does not
depend on the database, and tests can replace these functions with fakes.
"""
import os
import uuid
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "trailpeak_docs"
DENSE, SPARSE = "dense", "bm25"            # the two vector names inside each point
DENSE_SIZE = 1536                          # text-embedding-3-small
BM25_MODEL = "Qdrant/bm25"
BM25_AVG_LEN = 94.2                        # average words per chunk after stopwords (measured on data/, 20 Nov 2026)
RRF_K = 60                                 # set explicitly: Qdrant's own default is 2


@lru_cache(maxsize=1)
def get_client():
    """One client per process, reused for every question (it keeps its connections open)."""
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def _bm25():
    """FastEmbed's BM25 model: k1 = 1.2 and b = 0.75 (its defaults), avg_len measured on our chunks.
    The same model makes the chunk vectors (embed) and the question vectors (query_embed)."""
    return SparseTextEmbedding(BM25_MODEL, avg_len=BM25_AVG_LEN)


def _sparse(embedding):
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def point_id(chunk_id):
    """An md5 hex string has exactly the 32 hex digits of a UUID, so it converts one to one."""
    return str(uuid.UUID(chunk_id))


def recreate_collection():
    """Delete the collection if it exists and create it empty, with a dense and a sparse (BM25) vector."""
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config={DENSE: models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def upsert_chunks(chunks):
    """Store chunks, each a dict with id, text, source, position and embedding (the dense vector)."""
    sparse = list(_bm25().embed([chunk["text"] for chunk in chunks]))
    points = [
        models.PointStruct(
            id=point_id(chunk["id"]),
            vector={DENSE: chunk["embedding"], SPARSE: _sparse(vector)},
            payload={"text": chunk["text"], "source": chunk["source"], "position": chunk["position"]},
        )
        for chunk, vector in zip(chunks, sparse)
    ]
    get_client().upsert(COLLECTION_NAME, points=points, wait=True)


def count():
    return get_client().count(COLLECTION_NAME, exact=True).count


def _hit(point, with_distance=True):
    """A search result as a plain dict. Qdrant's cosine score is similarity (1 = identical); distance = 1 - score."""
    return {"id": str(point.id), "text": point.payload["text"], "source": point.payload["source"],
            "position": point.payload["position"], "distance": 1 - point.score if with_distance else None}


def dense_search(embedding, k):
    """The k chunks closest by meaning, best first."""
    points = get_client().query_points(COLLECTION_NAME, query=embedding, using=DENSE, limit=k, with_payload=True).points
    return [_hit(point) for point in points]


def hybrid_search(embedding, question, k, pool):
    """Meaning and keywords, merged inside Qdrant, in one round trip with two queries:
      1. the closest chunk by meaning (its distance is what the refusal gate checks);
      2. the top `pool` by meaning and the top `pool` by BM25, fused with RRF, keeping the first k.
    Returns (hits, closest_distance). A fused hit has no distance: an RRF score is not a distance."""
    question_sparse = _sparse(next(iter(_bm25().query_embed(question))))
    closest, fused = get_client().query_batch_points(COLLECTION_NAME, requests=[
        models.QueryRequest(query=embedding, using=DENSE, limit=1, with_payload=True),
        models.QueryRequest(
            prefetch=[models.Prefetch(query=embedding, using=DENSE, limit=pool),
                      models.Prefetch(query=question_sparse, using=SPARSE, limit=pool)],
            query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
            limit=k,
            with_payload=True,
        ),
    ])
    closest_distance = 1 - closest.points[0].score if closest.points else float("inf")
    return [_hit(point, with_distance=False) for point in fused.points], closest_distance


def all_chunks():
    """Every stored chunk (id, text, source, position), read page by page."""
    chunks, offset = [], None
    while True:
        points, offset = get_client().scroll(COLLECTION_NAME, limit=256, offset=offset,
                                             with_payload=True, with_vectors=False)
        chunks += [{"id": str(p.id), "text": p.payload["text"], "source": p.payload["source"],
                    "position": p.payload["position"]} for p in points]
        if offset is None:
            return chunks
