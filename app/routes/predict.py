from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import PredictRequest, PredictResponse
from app.services.sentiment_service import analyze_sentiment
from app.services.upload_parsing import parse_texts_from_upload

router = APIRouter(tags=["predict"])  


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    try:
        sentiments = analyze_sentiment(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment model error: {e}.")
    return PredictResponse(sentiments=sentiments)


@router.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)):
    content = await file.read()
    texts = parse_texts_from_upload(file.filename, content)
    if not texts:
        raise HTTPException(status_code=400, detail="No valid texts found in uploaded file")
    try:
        sentiments = analyze_sentiment(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment model error: {e}.")
    return PredictResponse(sentiments=sentiments)
