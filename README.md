# Experiment 7 – Customer Support RAG as a FastAPI service

The Experiment 6 RAG pipeline (PDF → chunks → MiniLM embeddings → FAISS → LangGraph
classify/retrieve/draft/review/retry/fallback) exposed through FastAPI, with input
validation, error handling, logging and response-time measurement.

```
experiment7/
├── app.py              # FastAPI: /health, /ask, validation, error handling, logging, timing
├── support_rag.py      # Expt 6 RAG pipeline + pluggable LLM backend (ollama | cloud | mock)
├── test_api.py         # Runs the testing checklist, writes observation_table.csv
├── static/index.html   # Simple chat UI served at /
├── Dockerfile          # Container image (model + FAISS index baked in)
├── render.yaml         # Render free-plan deployment blueprint
├── .dockerignore
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
First start downloads the embedding model (~90 MB, via fastembed) and builds the FAISS index into
`faiss_index/`; later starts load the cached index. Wait for `startup ok` in the console.

Open **http://127.0.0.1:8000/** for the chat UI, or **http://127.0.0.1:8000/docs** → `POST /ask` → *Try it out*:
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

## 5. Docker (local)

Requires Docker Desktop.
```bash
docker build -t support-rag .
# Ollama running on your PC (the container reaches the host via host.docker.internal):
docker run --rm -p 8080:8080 -e LLM_BACKEND=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 support-rag
# or no LLM at all:
docker run --rm -p 8080:8080 -e LLM_BACKEND=mock support-rag
```
Open http://localhost:8080. `.env` is excluded from the image, so pass config with `-e` or `--env-file .env`.

## 6. Deploy to Render (free, no card)

Render's free plan gives 512 MB RAM. The app uses ~280 MB because embeddings run through
fastembed (ONNX) rather than PyTorch, and the FAISS index is built into the image.

1. Push this repo to GitHub (`.env` is git-ignored, so your keys stay local):
   ```bash
   git push origin main
   ```
2. Sign up at https://render.com with your GitHub account (no card needed for the free plan).
3. **New → Blueprint**, pick this repo. Render reads `render.yaml` and asks for:
   - `CLOUD_MODEL`: `openai/gpt-oss-20b` (also add `MAX_NEW_TOKENS` = `1024`: it reasons before answering)
   - `CLOUD_API_KEY`: a free Groq API key from https://console.groq.com (stored as a secret)

   (Hugging Face's free inference credits are tiny and return HTTP 402 once used up, so the
   blueprint points `CLOUD_BASE_URL` at Groq's OpenAI-compatible API instead.)

   No token? After the first deploy, open the service's **Environment** tab and set `LLM_BACKEND=mock`.
4. The first build takes ~5–10 min. Then open `https://support-rag-xxxx.onrender.com` for the UI,
   `/docs` for Swagger, `/health?deep=true` to check the LLM backend. Logs are on the service's **Logs** tab.

Resilience (cloud backend): the app checks the provider's model list at startup and switches
automatically if `CLOUD_MODEL` has been retired (the log says which model it picked). Rate limits,
timeouts and provider errors are retried once. With `EXTRACTIVE_FALLBACK=true` (set in `render.yaml`),
if the LLM still fails (e.g. free quota used up) the answer is the best-matching FAQ passage, tagged
"LLM offline" in the UI, instead of an error. Locally it defaults to `false`, so the backend-down test still sees 503.

Free-plan notes: the service sleeps after ~15 min without traffic, and the first request after
that takes ~1 min to wake it. Ollama can't run on the free plan, so use `cloud` or `mock`.

Run the checklist against it: `API_URL=https://support-rag-xxxx.onrender.com python test_api.py`
(PowerShell: `$env:API_URL="https://..."; python test_api.py`).
