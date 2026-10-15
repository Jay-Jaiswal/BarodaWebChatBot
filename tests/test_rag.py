import pytest

from rag import (
    ABSTENTION_MESSAGE,
    GeminiConfigurationError,
    answer_question,
    ask_knowledge_base,
    generate_answer,
    load_system_prompt,
)
from ingestion import (
    DuplicateDocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    chunk_text,
    extract_document,
    extract_pdf_text,
    get_document_by_hash,
    process_document,
)
from vector_store import index_document, retrieve


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("knowledge" in lower),
                    float("python" in lower),
                    float("employee" in lower),
                ]
            )
        return vectors


def test_system_prompt_loads() -> None:
    configuration = load_system_prompt()
    assert configuration["role"]
    assert configuration["citation_requirements"]


def test_missing_context_is_explicit() -> None:
    assert "not available" in answer_question("What is this about?").lower()


def test_generate_answer_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        generate_answer("Say hello.")


def test_txt_and_markdown_extraction(tmp_path) -> None:
    text_path = tmp_path / "notes.txt"
    markdown_path = tmp_path / "guide.md"
    text_path.write_text("Plain text heading\nDetails.", encoding="utf-8")
    markdown_path.write_text("# Heading\n\nMarkdown details.", encoding="utf-8")

    text_pages = extract_document(text_path)
    markdown_pages = extract_document(markdown_path)
    assert text_pages[0]["text"].startswith("Plain text heading")
    assert "# Heading" in markdown_pages[0]["text"]
    assert markdown_pages[0]["metadata"]["page"] == -1


def test_pdf_extraction_preserves_page_metadata(tmp_path) -> None:
    import fitz

    pdf_path = tmp_path / "manual.pdf"
    with fitz.open() as document:
        for content in ("first page", "second page"):
            page = document.new_page()
            page.insert_text((72, 72), content)
        document.save(pdf_path)

    pages = extract_pdf_text(pdf_path)
    assert [page["metadata"]["page"] for page in pages] == [1, 2]
    assert pages[1]["text"].strip() == "second page"


def test_chunk_creation_has_overlap() -> None:
    chunks = chunk_text("a" * 100, chunk_size=40, overlap=10)
    assert len(chunks) == 3
    assert chunks[0][-10:] == chunks[1][:10]


def test_empty_and_unsupported_documents_are_rejected(tmp_path) -> None:
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        extract_document(empty_path)

    unsupported_path = tmp_path / "image.csv"
    unsupported_path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        extract_document(unsupported_path)


def test_duplicate_document_detection(tmp_path) -> None:
    db_path = tmp_path / "documents.db"
    source = tmp_path / "source.txt"
    source.write_text("same content", encoding="utf-8")
    process_document(source, db_path)
    with pytest.raises(DuplicateDocumentError):
        process_document(source, db_path)


def test_persistent_indexing_retrieval_and_source_metadata(tmp_path) -> None:
    source = tmp_path / "knowledge.txt"
    source.write_text(
        "Knowledge base passage: Python is used for data analysis.\n"
        "This sentence identifies the source passage.",
        encoding="utf-8",
    )
    document = index_document(
        source,
        persist_directory=tmp_path / "chroma",
        db_path=tmp_path / "documents.db",
        collection_name="test_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    matches = retrieve(
        "What is the knowledge base passage about?",
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    assert matches
    assert matches[0]["metadata"]["document_id"] == document["document_id"]
    assert matches[0]["metadata"]["filename"] == "knowledge.txt"
    assert "Python is used" in matches[0]["text"]
    status = get_document_by_hash(document["content_hash"], tmp_path / "documents.db")
    assert status["status"] == "indexed"


def test_known_answer_has_valid_citations_and_excerpts(tmp_path, monkeypatch) -> None:
    source = tmp_path / "policy.txt"
    source.write_text(
        "The support team responds to customer requests within two business days.",
        encoding="utf-8",
    )
    document = index_document(
        source,
        persist_directory=tmp_path / "chroma",
        db_path=tmp_path / "documents.db",
        collection_name="rag_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "rag.generate_answer",
        lambda prompt: "The support team responds within two business days.",
    )
    result = ask_knowledge_base(
        "How quickly does the support team respond?",
        persist_directory=tmp_path / "chroma",
        collection_name="rag_collection",
        embedding_model=FakeEmbeddingModel(),
        relevance_threshold=1.0,
    )
    assert result.answer.startswith("The support team")
    assert result.citations[0].document_id == document["document_id"]
    assert result.citations[0].filename == "policy.txt"
    assert result.citations[0].chunk_id
    assert "two business days" in result.supporting_excerpts[0]["text"]


def test_missing_and_irrelevant_questions_abstain(tmp_path) -> None:
    source = tmp_path / "facts.txt"
    source.write_text("The office closes at 5 PM.", encoding="utf-8")
    index_document(
        source,
        persist_directory=tmp_path / "chroma",
        db_path=tmp_path / "documents.db",
        collection_name="abstain_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    missing = ask_knowledge_base(
        "What is the employee salary?",
        persist_directory=tmp_path / "empty-chroma",
        collection_name="abstain_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    assert missing.abstained
    assert missing.answer == ABSTENTION_MESSAGE

    irrelevant = ask_knowledge_base(
        "What is the employee salary?",
        persist_directory=tmp_path / "chroma",
        collection_name="abstain_collection",
        embedding_model=FakeEmbeddingModel(),
        relevance_threshold=0.01,
    )
    assert irrelevant.abstained
    assert irrelevant.citations == []


def test_default_threshold_keeps_known_evidence(tmp_path, monkeypatch) -> None:
    source = tmp_path / "known.txt"
    source.write_text("Python is used for data analysis.", encoding="utf-8")
    index_document(
        source,
        persist_directory=tmp_path / "chroma",
        db_path=tmp_path / "documents.db",
        collection_name="default_threshold_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
    matches = retrieve(
        "What is Python used for?",
        persist_directory=tmp_path / "chroma",
        collection_name="default_threshold_collection",
        embedding_model=FakeEmbeddingModel(),
    )
    assert matches
