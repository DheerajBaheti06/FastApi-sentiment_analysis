from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.analyze import router as analyze_router
from app.routes.predict import router as predict_router
from app.routes.wordcloud import router as wordcloud_router

app = FastAPI(title="SIH Sentiment Service", version="1.0.0")

# CORS for easy frontend integration (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(wordcloud_router)
app.include_router(predict_router)


@app.get("/")
def root():
    return {"service": "SIH Sentiment Service", "version": "1.0.0"}
