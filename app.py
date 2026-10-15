"""Streamlit UI for document indexing and retrieval-backed Gemini chat."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ingestion import (
    DocumentProcessingError,
    DuplicateDocumentError,
    list_documents,
)
from rag import (
    GeminiConfigurationError,
    GeminiGenerationError,
    ask_knowledge_base,
)
from vector_store import get_document_chunk_count, index_document

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma"
STATUS_DB = DATA_DIR / "documents.db"
SUPPORTED_TYPES = ["pdf", "txt", "md", "markdown"]


def _save_uploaded_file(uploaded_file: Any) -> Path:
    """Write an upload to a temporary file while preserving its extension."""
    suffix = Path(uploaded_file.name).suffix.lower()
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, prefix="knowledge-base-", delete=False
    )
    try:
        temporary.write(uploaded_file.getbuffer())
    finally:
        temporary.close()
    return Path(temporary.name)


def _display_documents() -> None:
    records = list_documents(STATUS_DB)
    st.subheader("Indexed documents")
    if not records:
        st.caption("No documents have been indexed yet.")
        return

    rows = []
    for record in records:
        rows.append(
            {
                "Filename": record["filename"],
                "Status": record["status"],
                "Chunks": get_document_chunk_count(
                    record["document_id"],
                    persist_directory=CHROMA_DIR,
                ),
                "Created": record["created_at"],
                "Error": record["error_message"] or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _process_uploads(uploaded_files: list[Any]) -> None:
    for uploaded_file in uploaded_files:
        temporary_path: Path | None = None
        try:
            temporary_path = _save_uploaded_file(uploaded_file)
            document = index_document(
                temporary_path,
                persist_directory=CHROMA_DIR,
                db_path=STATUS_DB,
                display_filename=uploaded_file.name,
            )
            st.success(
                f"{uploaded_file.name}: indexed {len(document['chunks'])} chunk(s)."
            )
        except DuplicateDocumentError as error:
            st.info(f"{uploaded_file.name}: already indexed. {error}")
        except DocumentProcessingError as error:
            st.error(f"{uploaded_file.name}: processing failed. {error}")
        except Exception:
            st.exception(
                f"{uploaded_file.name}: indexing failed unexpectedly. "
                "The document was not marked indexed."
            )
        finally:
            if temporary_path:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _chat() -> None:
    st.subheader("Ask the knowledge base")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("citations"):
                with st.expander("Sources and supporting excerpts"):
                    st.markdown("**Citations**")
                    for citation in message["citations"]:
                        st.caption(f"{citation['label']} | chunk {citation['chunk_id']}")
                    st.markdown("**Supporting excerpts**")
                    for excerpt in message["excerpts"]:
                        metadata = excerpt["metadata"]
                        page = metadata.get("page", -1)
                        source = metadata.get("filename", "unknown source")
                        if page != -1:
                            source += f", page {page}"
                        st.caption(
                            f"{source} | similarity {excerpt['similarity']:.3f} "
                            f"(distance {excerpt['distance']:.3f})"
                        )
                        st.info(excerpt["text"])

    question = st.chat_input("Ask about your indexed knowledge base")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        result = ask_knowledge_base(
            question,
            persist_directory=CHROMA_DIR,
            top_k=int(os.getenv("TOP_K", "5")),
        )
        answer = result.answer
        citations = [
            {"label": citation.label, "chunk_id": citation.chunk_id}
            for citation in result.citations
        ]
        excerpts = result.supporting_excerpts
    except (GeminiConfigurationError, GeminiGenerationError, ValueError) as error:
        answer, citations, excerpts = str(error), [], []
    except Exception:
        answer, citations, excerpts = (
            "The question could not be answered.",
            [],
            [],
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "excerpts": excerpts,
        }
    )
    st.rerun()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Knowledge Base Chatbot", page_icon="📚")
    st.title("AI Knowledge Base Chatbot")
    st.caption("Upload documents, persist their local index, and ask grounded questions.")

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or Markdown files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("Process uploads", type="primary"):
            _process_uploads(uploaded_files)

    _display_documents()
    _chat()


if __name__ == "__main__":
    main()
