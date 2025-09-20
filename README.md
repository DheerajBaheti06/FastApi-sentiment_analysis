# FastApi-sentiment_analysis

An ML service to analyze comments and provide per-comment sentiments, an overall summary, and a wordcloud with positive, negative, and neutral terms.

## About

Built with FastAPI using FinBERT (and optional SVM/TF-IDF). Exposes JSON and file-upload endpoints, generates a brief extractive summary, and can return a cached wordcloud image.

## Setup

Install dependencies (recommend a virtual environment):

```powershell
pip install -r requirements.txt
```

Place model files in either project root or `app/models`:

- `FINBERT_FINAL.BIN` (required)
- `SVM_FINAL.PKL` (optional)
- `TFIDF_VECTORIZER_FINAL.PKL` (optional)

## CLI usage

Run demo:

```powershell
python predictor.py
```

Single text:

```powershell
python predictor.py --text "This is great!"
```

Interactive loop:

```powershell
python predictor.py --interactive
```

Custom model paths:

```powershell
python predictor.py --finbert "e:\Dheeraj\Sentiment Analysis ml model files\Sentiment Analysis\FINBERT_FINAL.BIN" --svm "e:\Dheeraj\Sentiment Analysis ml model files\Sentiment Analysis\SVM_FINAL.PKL" --tfidf "e:\Dheeraj\Sentiment Analysis ml model files\Sentiment Analysis\TFIDF_VECTORIZER_FINAL.PKL"
```

## FastAPI Service (app/ structure)

Optional environment variables to override model locations:

- `FINBERT_PATH` — path to `FINBERT_FINAL.BIN`
- `SVM_PATH` — path to `SVM_FINAL.PKL`
- `TFIDF_PATH` — path to `TFIDF_VECTORIZER_FINAL.PKL`

If env vars are not provided, the service will autodiscover model files from `app/models` or project root.

Run with the helper script (Windows):

```powershell
./run_api.ps1 -BindHost 127.0.0.1 -Port 8000
```

Or run manually with uvicorn:

```powershell
$env:FINBERT_PATH = "e:\Dheeraj\Sentiment Analysis ml model files\Sentiment Analysis\FINBERT_FINAL.BIN"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Analyze (compact: sentiment + summary + wordcloud):

```powershell
curl -X POST http://127.0.0.1:8000/analyze/compact -H "Content-Type: application/json" -d '{
  "texts": [
    "The recent amendments regarding startup compliance will definitely boost investor confidence.",
    "While we appreciate the intent to simplify, the new compliance rules create a significant burden for MSMEs."
  ],
  "max_summary_sentences": 2,
  "include_wordcloud": true
}'
```

Response includes:

- `sentiments`: array per input text
- `summary`: short extractive summary
- `percentages`: positive/negative/neutral percentages
- `wordcloud`: optional object `{ "url": "/wordcloud/{key}", "key": "..." }`

Additional endpoints:

- `GET /wordcloud?text=...` → returns a JPEG image directly (204 if no content)
- `POST /wordcloud/generate` with `{ "texts": [ ... ] }` → returns `{ url, key }`
- `GET /wordcloud/{key}` → returns a cached JPEG by short key

Uploads:

- `POST /predict/upload` (form-data): `file=@comments.json|.csv|.txt`
- `POST /analyze/compact/upload` (form-data): `file=@...`, optional fields `max_summary_sentences`, `include_wordcloud`

Postman:

- Import `Sentiment_API.postman_collection.json`
- Set `baseUrl` to `127.0.0.1:8000` (or your chosen host/port)

## Deployment

### Docker (recommended)

```powershell
docker compose build
docker compose up -d
Start-Process http://127.0.0.1:8000/docs
```

Notes:

- The image copies `FINBERT_FINAL.BIN`, `SVM_FINAL.PKL`, and `TFIDF_VECTORIZER_FINAL.PKL` from the project root (if present).
- Port 8000 is exposed; change mapping in `compose.yaml` if needed.

### Windows service

```powershell
nssm install SentimentAPI "powershell.exe" "-ExecutionPolicy Bypass -File `"$(Resolve-Path .\run_api.ps1)`" -BindHost 0.0.0.0 -Port 8000
nssm start SentimentAPI
```

### Cloud

- Railway/Azure Container Apps/AWS ECS (Fargate)/Render can deploy with the Dockerfile. Health check: `/health`.
- To prefetch the tokenizer in the image:

```dockerfile
RUN python - <<'PY'\
from transformers import AutoTokenizer\
AutoTokenizer.from_pretrained('ProsusAI/finbert')\
PY
```
