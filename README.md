# Production RAG System

Production-ready Retrieval-Augmented Generation system built with FastAPI, LangChain, ChromaDB, OpenAI or Gemini, sentence-transformers, SQLite, and MLflow.

## Features

- Ingest real `.pdf`, `.txt`, and `.md` files through `POST /ingest`
- Chunk documents with recursive splitting (`512` size, `64` overlap)
- Persist embeddings in local ChromaDB
- Retrieve relevant chunks with similarity scores
- Generate grounded answers with `gpt-4o-mini` or Gemini
- Detect hallucinations with `cross-encoder/nli-deberta-v3-small`
- Retry once with a stricter prompt when support is weak
- Store query logs, hallucination logs, and feedback in SQLite
- Inject corrective user feedback back into retrieval as high-priority memory
- Track each query run in MLflow experiment `rag_production`
- Bonus streaming endpoint with SSE at `POST /query/stream`
- Streamlit frontend for ingestion, querying, feedback, and tracking

## Project Structure

```text
.
├── api/
│   ├── dependencies.py
│   ├── main.py
│   ├── routes_feedback.py
│   ├── routes_ingestion.py
│   ├── routes_query.py
│   └── routes_tracking.py
├── core/
│   ├── config.py
│   ├── database.py
│   ├── llm.py
│   ├── models.py
│   ├── schemas.py
│   ├── utils.py
│   └── vector_store.py
├── feedback/
│   └── service.py
├── hallucination/
│   └── service.py
├── ingestion/
│   └── service.py
├── retrieval/
│   └── service.py
├── tracking/
│   └── service.py
├── streamlit_app.py
├── data/
│   └── uploads/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3.11.14 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Choose an LLM provider in `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Or:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
```

### 4. Run the FastAPI server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

If you prefer a tracked server URI, update `MLFLOW_TRACKING_URI` in `.env` to `http://127.0.0.1:5000`.

### 6. Run the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

## API Usage

### Ingest documents

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/absolute/path/to/file1.pdf" \
  -F "files=@/absolute/path/to/file2.md"
```

### Query documents

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key findings from the ingested documents",
    "top_k": 4
  }'
```

### Streaming query

```bash
curl -N -X POST "http://127.0.0.1:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main conclusions?",
    "top_k": 4
  }'
```

### Submit feedback

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "paste-query-id-here",
    "rating": 1,
    "correction": "The correct policy effective date is 2026-01-15."
  }'
```

### MLflow summary

```bash
curl "http://127.0.0.1:8000/mlflow/summary"
```

## Streamlit UI

The Streamlit app provides:

- Multi-file upload for `.pdf`, `.txt`, and `.md`
- Standard and streaming query modes
- Source and score inspection
- Inline feedback submission with optional correction
- MLflow experiment summary view

The frontend reads these optional `.env` values:

```bash
STREAMLIT_API_BASE_URL=http://127.0.0.1:8000
MLFLOW_UI_URL=http://127.0.0.1:5000
```

## Docker

### Start app + MLflow

```bash
docker compose up --build
```

FastAPI will be available at `http://127.0.0.1:8000` and MLflow at `http://127.0.0.1:5000`.

## Notes

- The first start downloads the embedding and hallucination models from Hugging Face.
- `LLM_PROVIDER` supports `openai` and `gemini`.
- Query responses include `query_id`, which is required for the feedback loop.
- Corrective feedback with rating `<= 2` is embedded into Chroma with elevated priority so future retrieval is biased toward the correction.
- SQLite data is stored in `data/rag_app.db`.
