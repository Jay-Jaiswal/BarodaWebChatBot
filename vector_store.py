"""Sentence Transformer embeddings and persistent ChromaDB retrieval."""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

from ingestion import (
    DuplicateDocumentError,
    mark_document_failed,
    mark_document_indexed,
    process_document,
)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "knowledge_base"
# Starting point for this corpus; calibrate for other document collections.
DEFAULT_RELEVANCE_THRESHOLD = 0.70


def _embedding_rows(value: Any) -> list[list[float]]:
    """Normalize numpy and simple test-double embedding outputs."""
    return value.tolist() if hasattr(value, "tolist") else value


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load and cache the local Sentence Transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=8)
def _get_chroma_client(persist_directory: str):
    """Reuse one Chroma client per directory within the Streamlit process."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )


def create_collection(
    persist_directory: str = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Create or reopen a persistent cosine-distance ChromaDB collection."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    client = _get_chroma_client(str(directory.resolve()))
    return client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_document_chunk_count(
    document_id: str,
    persist_directory: str = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """Count persisted chunks for a document without loading embeddings."""
    collection = create_collection(persist_directory, collection_name)
    if collection.count() == 0:
        return 0
    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    return len(result.get("ids", []))


def index_document(
    path: str | Path,
    persist_directory: str = "data/chroma",
    db_path: str | Path = "data/documents.db",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: Any | None = None,
    display_filename: str | None = None,
) -> dict[str, Any]:
    """Process, embed, and persist a document, avoiding duplicate content."""
    collection = create_collection(persist_directory, collection_name)
    document = process_document(path, db_path, display_filename=display_filename)
    try:
        model = embedding_model or get_embedding_model(
            os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        texts = [chunk["text"] for chunk in document["chunks"]]
        embeddings = _embedding_rows(model.encode(texts, normalize_embeddings=True))
        collection.add(
            ids=[chunk["id"] for chunk in document["chunks"]],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {**chunk["metadata"], "chunk_id": chunk["id"]}
                for chunk in document["chunks"]
            ],
        )
        mark_document_indexed(document, db_path)
    except Exception:
        mark_document_failed(
            document,
            db_path,
            "Embedding or ChromaDB indexing failed.",
        )
        raise
    return document


def retrieve(
    question: str,
    persist_directory: str = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    relevance_threshold: float | None = None,
    embedding_model: Any | None = None,
) -> list[dict[str, Any]]:
    """Return relevant chunks with cosine distance, similarity, and source metadata.

    ``relevance_threshold`` is an optional cosine-distance cutoff and requires
    calibration on representative questions; no universal value is assumed.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if relevance_threshold is None:
        configured_threshold = os.getenv("RELEVANCE_THRESHOLD")
        try:
            relevance_threshold = (
                float(configured_threshold)
                if configured_threshold
                else DEFAULT_RELEVANCE_THRESHOLD
            )
        except ValueError as error:
            raise ValueError("RELEVANCE_THRESHOLD must be a number.") from error

    collection = create_collection(persist_directory, collection_name)
    if collection.count() == 0:
        return []
    model = embedding_model or get_embedding_model(
        os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    query_embedding = _embedding_rows(
        model.encode([question], normalize_embeddings=True)
    )
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    matches: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if relevance_threshold is not None and distance > relevance_threshold:
            continue
        metadata = {**metadata, "chunk_id": metadata.get("chunk_id", chunk_id)}
        matches.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1.0 - distance,
            }
        )
    return matches


__all__ = [
    "DuplicateDocumentError",
    "create_collection",
    "get_document_chunk_count",
    "get_embedding_model",
    "index_document",
    "retrieve",
]
