import os
from fastapi import APIRouter

router = APIRouter(prefix="/debug", tags=["debug"])


def _stat(path: str):
    try:
        if os.path.exists(path):
            return {
                "exists": True,
                "size_bytes": os.path.getsize(path),
            }
        return {"exists": False, "size_bytes": 0}
    except Exception as e:
        return {"exists": False, "error": str(e)}


@router.get("/status")
async def status():
    base = "/app"
    finbert = os.getenv("FINBERT_PATH") or os.path.join(base, "FINBERT_FINAL.BIN")
    svm = os.getenv("SVM_PATH") or os.path.join(base, "SVM_FINAL.PKL")
    tfidf = os.getenv("TFIDF_PATH") or os.path.join(base, "TFIDF_VECTORIZER_FINAL.PKL")

    return {
        "env": {
            "PORT": os.getenv("PORT"),
            "BATCH_SIZE": os.getenv("BATCH_SIZE"),
            "MAX_LENGTH": os.getenv("MAX_LENGTH"),
            "TORCH_NUM_THREADS": os.getenv("TORCH_NUM_THREADS"),
            "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS"),
            "TRUSTED_HOSTS": os.getenv("TRUSTED_HOSTS"),
            "API_KEY_set": bool(os.getenv("API_KEY")),
        },
        "files": {
            "FINBERT_FINAL.BIN": _stat(finbert),
            "SVM_FINAL.PKL": _stat(svm),
            "TFIDF_VECTORIZER_FINAL.PKL": _stat(tfidf),
        },
    }
