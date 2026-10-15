# AI Knowledge Base Chatbot

A beginner-friendly prototype of a Retrieval-Augmented Generation (RAG) chatbot.
Users will upload documents, retrieve relevant evidence, and receive answers with
source citations. The application is designed to treat uploaded content and user
messages as untrusted input.

## Objective

- Extract text from PDFs with page metadata.
- Embed and persist document chunks using `all-MiniLM-L6-v2` and ChromaDB.
- Generate answers with Google Gemini through the official `google-genai` SDK.
- Refuse to invent answers when the knowledge base lacks evidence.
- Resist prompt injection from documents and user messages.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe` directly.

## Run

```powershell
streamlit run app.py
```

The current foundation includes importable modules and configuration; the
ingestion, retrieval, and LLM response flow will be wired in the next step.

## Architecture

1. `app.py` - Streamlit user interface.
2. `ingestion.py` - PDF extraction while preserving page numbers.
3. `vector_store.py` - persistent ChromaDB collection.
4. `rag.py` - prompt configuration and RAG orchestration.
5. `prompts/system_prompt.yaml` - editable behavior and safety rules.
6. `data/` - local persistent application data.
7. `sample_documents/` - optional documents for demos.

## Ingest and retrieve documents

The ingestion and retrieval APIs can be used before the Streamlit UI is wired
to them:

```powershell
.\.venv\Scripts\python.exe -c "from vector_store import index_document, retrieve; d=index_document('sample_documents\\example.pdf'); print(d['document_id'], len(d['chunks'])); print(retrieve('What is this document about?'))"
```

Supported formats are PDF, TXT, and Markdown. Chunks use approximately 2,000
characters with 300 characters of overlap (a simple approximation to 500 and
75 tokens). ChromaDB is persisted under `data/chroma/`, while processing
status and content hashes are stored in `data/documents.db`. Retrieval returns
the source filename, page number when available, chunk index, cosine distance,
and a derived similarity value. The optional distance relevance threshold must
be calibrated against representative questions rather than treated as a
universal constant.

## Gemini configuration

Set `LLM_PROVIDER=gemini` and provide a Gemini API key in `.env`:

```powershell
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.8-flash
```

The model name is configurable because model availability depends on the API
key and Google account. Verify a chosen model with a simple SDK request before
using it in the RAG flow. Never place the key in source code, browser code, or
Git. The local extraction, embeddings, ChromaDB storage, and citation metadata
remain local; only the assembled prompt is sent to Gemini.

## RAG answers and citations

Questions are embedded locally and matched against up to five persistent
ChromaDB chunks. `RELEVANCE_THRESHOLD` defaults to `0.70` for the current
prototype corpus as a starting cosine-distance cutoff. It is configurable and must be calibrated against
representative questions and documents; no threshold is universally correct.

The structured RAG response contains the Gemini answer, supporting excerpts,
and citations generated from stored document ID, chunk ID, filename, and page
metadata. If no retrieved evidence passes the threshold, the chatbot abstains
instead of guessing. Retrieved document text is delimited as untrusted evidence,
so instructions inside documents cannot override the system prompt.

Run the smoke test:

```powershell
.\.venv\Scripts\python.exe test_gemini.py
```

It reports a clear skip when credentials are not configured and otherwise
performs one real Gemini request, which also verifies that `GEMINI_MODEL` is
available to the configured API key.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
