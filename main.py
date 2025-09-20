import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from predictor import HybridSentimentPredictor

app = FastAPI(title="SIH Sentiment Analysis API", version="1.0.0")

predictor: Optional[HybridSentimentPredictor] = None

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    sentiments: List[str]

@app.on_event("startup")
def load_models():
    global predictor
    finbert_path = os.getenv("FINBERT_PATH")
    svm_path = os.getenv("SVM_PATH")
    tfidf_path = os.getenv("TFIDF_PATH")
    predictor = HybridSentimentPredictor(finbert_path, svm_path, tfidf_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    sentiments = predictor.predict(req.texts)
    return PredictResponse(sentiments=sentiments)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
