"""Local document extraction, chunking, hashing, and processing status."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
CHUNK_SIZE = 2_000
CHUNK_OVERLAP = 300


class DocumentProcessingError(RuntimeError):
    """Base error for document processing failures."""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when a document extension is not supported."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no extractable text."""


class DuplicateDocumentError(DocumentProcessingError):
    """Raised when the same document content was already processed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_status_db(db_path: str | Path = "data/documents.db") -> None:
    """Create the document status table if it does not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('processing', 'indexed', 'failed')),
                created_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        connection.commit()


def get_document_by_hash(
    content_hash: str, db_path: str | Path = "data/documents.db"
) -> dict[str, Any] | None:
    initialize_status_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
    return dict(row) if row else None


def list_documents(db_path: str | Path = "data/documents.db") -> list[dict[str, Any]]:
    """Return document processing records for display after app reruns."""
    initialize_status_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def _set_status(
    document_id: str,
    filename: str,
    content_hash: str,
    status: str,
    db_path: str | Path,
    error_message: str | None = None,
) -> None:
    initialize_status_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO documents
                (document_id, filename, content_hash, status, created_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                status = excluded.status,
                error_message = excluded.error_message
            """,
            (
                document_id,
                filename,
                content_hash,
                status,
                _utc_now(),
                error_message,
            ),
        )
        connection.commit()


def _read_source_pages(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz

        try:
            with fitz.open(path) as document:
                pages = [
                    {"text": page.get_text("text"), "page_number": page_number}
                    for page_number, page in enumerate(document, start=1)
                ]
        except Exception as error:
            raise DocumentProcessingError(f"Could not read PDF '{path.name}': {error}") from error
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise DocumentProcessingError(
                f"'{path.name}' is not valid UTF-8 text."
            ) from error
        except OSError as error:
            raise DocumentProcessingError(f"Could not read '{path.name}': {error}") from error
        pages = [{"text": text, "page_number": None}]

    if not any(page["text"].strip() for page in pages):
        raise EmptyDocumentError(f"'{path.name}' contains no extractable text.")
    return pages


def extract_pdf_text(path: str | Path) -> list[dict[str, Any]]:
    """Extract PDF pages while preserving the original one-based page number."""
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise UnsupportedFileTypeError("extract_pdf_text requires a .pdf file.")
    return [
        {"text": page["text"], "metadata": {"source": pdf_path.name, "page": page["page_number"]}}
        for page in _read_source_pages(pdf_path)
    ]


def extract_document(path: str | Path) -> list[dict[str, Any]]:
    """Extract supported PDF, TXT, or Markdown content with source metadata."""
    source_path = Path(path)
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{source_path.suffix or '<none>'}'. Supported: {supported}."
        )
    if not source_path.is_file():
        raise DocumentProcessingError(f"Document does not exist: {source_path}")

    return [
        {
            "text": page["text"],
            "metadata": {
                "source": source_path.name,
                "filename": source_path.name,
                "page": page["page_number"] if page["page_number"] is not None else -1,
            },
        }
        for page in _read_source_pages(source_path)
    ]


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character chunks (roughly 500/75 tokens)."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def process_document(
    path: str | Path,
    db_path: str | Path = "data/documents.db",
    display_filename: str | None = None,
) -> dict[str, Any]:
    """Extract and chunk a document, recording processing status in SQLite."""
    source_path = Path(path)
    filename = display_filename or source_path.name
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type: {source_path.suffix or '<none>'}")
    if not source_path.is_file():
        raise DocumentProcessingError(f"Document does not exist: {source_path}")

    content_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    existing = get_document_by_hash(content_hash, db_path)
    if existing:
        raise DuplicateDocumentError(
            f"'{filename}' duplicates '{existing['filename']}' "
            f"(document_id={existing['document_id']})."
        )

    document_id = uuid.uuid4().hex
    _set_status(document_id, filename, content_hash, "processing", db_path)
    try:
        pages = extract_document(source_path)
        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        for page in pages:
            for text in chunk_text(page["text"]):
                chunks.append(
                    {
                        "id": f"{document_id}-{chunk_index}",
                        "text": text,
                        "metadata": {
                            "document_id": document_id,
                            "filename": filename,
                            "source": filename,
                            "page": page["metadata"]["page"],
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
        if not chunks:
            raise EmptyDocumentError(f"'{source_path.name}' contains no extractable text.")
    except Exception as error:
        _set_status(
            document_id,
            filename,
            content_hash,
            "failed",
            db_path,
            str(error),
        )
        raise

    return {
        "document_id": document_id,
        "filename": filename,
        "content_hash": content_hash,
        "chunks": chunks,
    }


def mark_document_indexed(
    document: dict[str, Any], db_path: str | Path = "data/documents.db"
) -> None:
    _set_status(
        document["document_id"],
        document["filename"],
        document["content_hash"],
        "indexed",
        db_path,
    )


def mark_document_failed(
    document: dict[str, Any],
    db_path: str | Path = "data/documents.db",
    error_message: str = "Document indexing failed.",
) -> None:
    _set_status(
        document["document_id"],
        document["filename"],
        document["content_hash"],
        "failed",
        db_path,
        error_message,
    )
