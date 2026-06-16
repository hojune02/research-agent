from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import DocumentChunk


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model once and cache it.

    all-MiniLM-L6-v2 is small enough for local CPU use and good enough
    for an MVP semantic search system.
    """
    return SentenceTransformer(settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Persistent Chroma client.

    Data is stored on disk at settings.CHROMA_PATH.
    """
    return chromadb.PersistentClient(path=settings.CHROMA_PATH)


def get_documents_collection() -> Collection:
    """
    Get or create the main documents collection.
    """
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={
            "description": "PaperOps document chunks for multi-user RAG",
        },
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert text chunks into dense vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def index_chunks(chunks: list[DocumentChunk]) -> int:
    """
    Add parsed document chunks to Chroma.

    Each chunk is stored with:
    - id
    - text document
    - embedding
    - metadata

    Metadata is essential for user/project filtering later.
    """
    if not chunks:
        return 0

    collection = get_documents_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        # Make ID globally unique across users/projects/files.
        global_id = f"{chunk.user_id}::{chunk.project_id}::{chunk.source}::{chunk.chunk_id}"

        ids.append(global_id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "user_id": chunk.user_id,
                "project_id": chunk.project_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "char_count": chunk.char_count,
            }
        )

    embeddings = embed_texts(documents)

    # Use upsert instead of add so re-indexing the same file does not crash
    # due to duplicate IDs.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def get_collection_count() -> int:
    """
    Total number of chunks in the collection.
    """
    collection = get_documents_collection()
    return collection.count()