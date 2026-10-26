# Experiment 7 – Customer Support RAG as a FastAPI service

The Experiment 6 RAG pipeline (PDF → chunks → MiniLM embeddings → FAISS → LangGraph
classify/retrieve/draft/review/retry/fallback) exposed through FastAPI, with input
validation, error handling, logging and response-time measurement.

```
experiment7/
├── app.py              # FastAPI: /health, /ask, validation, error handling, logging, timing
├── support_rag.py      # Expt 6 RAG pipeline + pluggable LLM backend (ollama | cloud | mock)
├── test_api.py         # Runs the testing checklist, writes observation_table.csv
├── requirements.txt
├── .env.example        # Copy to .env (secrets/config; never upload)
├── .gitignore
└── data/
    └── Expt6_Source_File.pdf   # knowledge base from Expt 6
```

## 1. Setup (once)

```bash
cd experiment7
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

## 2. Pick ONE backend (edit `.env`)

**Option B – Ollama (local, recommended)**
```bash
# install from https://ollama.com, then:
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b      # type a test message, then /bye
```
`.env`: `LLM_BACKEND=ollama`

**Option A – Cloud API (free/trial quota)**
Create a Hugging Face access token (or another OpenAI-compatible provider), then in `.env`:
```
LLM_BACKEND=cloud
CLOUD_BASE_URL=https://router.huggingface.co/v1
CLOUD_MODEL=<model id currently available on your free quota>
CLOUD_API_KEY=<your token>
```

**No model available?** `LLM_BACKEND=mock` tests the API structure without any LLM.

## 3. Run the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
First start downloads the embedding model (~90 MB) and builds the FAISS index into
`faiss_index/`; later starts load the cached index. Wait for `startup ok` in the console.

Open **http://127.0.0.1:8000/docs** → `POST /ask` → *Try it out*:
```json
{ "question": "How many titles can I borrow at a time with Kindle Unlimited?" }
```

curl equivalent:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"How do I update the software on my Fire TV?\"}"
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/health?deep=true"
```

## 4. Run the testing checklist

In a second terminal (API still running):
```bash
python test_api.py
```
Backend-failure test: stop Ollama (quit it from the tray / `sudo systemctl stop ollama`),
or set `OLLAMA_BASE_URL=http://localhost:9999` in `.env` and restart uvicorn, then:
```bash
python test_api.py --backend-down
```
Both runs write to `observation_table.csv`. Logs (with latency per request) are in `logs/app.log`.

## Error responses

| Situation | HTTP | Body `error` |
|---|---|---|
| Empty / > 1000-char question, missing field | 422 | `invalid_request` |
| LLM down, timeout, bad key, quota exceeded | 503 | `backend_unavailable` (details only in server log) |
| Knowledge base failed to load at startup | 503 | `service_unavailable` |
| Any other bug | 500 | `internal_error` |

Every response carries `X-Request-ID` and `X-Response-Time-ms` headers.
