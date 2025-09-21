import os
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.routes.health import router as health_router
from app.routes.analyze import router as analyze_router
from app.routes.predict import router as predict_router
from app.routes.wordcloud import router as wordcloud_router
from app.routes.debug import router as debug_router
from app.services.sentiment_service import get_predictor

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), override=False)
app = FastAPI(title="SIH Sentiment Service", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

trusted_hosts_env = os.getenv("TRUSTED_HOSTS")
if trusted_hosts_env:
    hosts = [h.strip() for h in trusted_hosts_env.split(",") if h.strip()]
    if hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

API_KEY = os.getenv("API_KEY")

def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Routers
app.include_router(health_router)
app.include_router(analyze_router, dependencies=[Depends(require_api_key)])
app.include_router(wordcloud_router, dependencies=[Depends(require_api_key)])
app.include_router(predict_router, dependencies=[Depends(require_api_key)])
app.include_router(debug_router)


@app.get("/")
def root():
    return {"service": "SIH Sentiment Service", "version": "1.0.0"}


@app.on_event("startup")
async def _log_startup():
    try:
        logger.info("Startup: PORT=%s HOST=%s", os.getenv("PORT"), os.getenv("HOST"))
        logger.info("Startup: ALLOWED_ORIGINS=%s", os.getenv("ALLOWED_ORIGINS", "*"))
        logger.info("Startup: TRUSTED_HOSTS=%s", os.getenv("TRUSTED_HOSTS", ""))
        logger.info("Startup: API_KEY set=%s", bool(os.getenv("API_KEY")))
        base = "/app"
        finbert = os.getenv("FINBERT_PATH") or os.path.join(base, "FINBERT_FINAL.BIN")
        logger.info("Startup: FINBERT_PATH=%s exists=%s", finbert, os.path.exists(finbert))
        # Warm-up: initialize predictor once to avoid first-request latency
        try:
            _ = get_predictor()
            logger.info("Startup: predictor initialized")
        except Exception as me:
            logger.error("Startup: predictor init failed: %s", me)
    except Exception as e:
        logger.error("Startup logging failed: %s", e)
