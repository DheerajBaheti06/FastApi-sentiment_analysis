from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from app.models.schemas import AnalyzeCompactRequest, AnalyzeCompactResponse, WordcloudRef
from app.services.sentiment_service import analyze_sentiment
from app.services.upload_parsing import parse_texts_from_upload

router = APIRouter(prefix="/analyze", tags=["analyze"]) 


@router.post("/compact", response_model=AnalyzeCompactResponse)
async def analyze_compact(req: AnalyzeCompactRequest, request: Request):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    try:
        sentiments = analyze_sentiment(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment model error: {e}.")

    # Overall summary: concatenate texts and summarize
    joined = " ".join([t for t in req.texts if t and t.strip()])
    from app.services.summary_service import summarize_text
    overall_summary = summarize_text(joined, max_sentences=req.max_summary_sentences)

    # Single wordcloud for all texts combined (optional)
    from app.services.wordcloud_service import generate_wordcloud_base64, cache_wordcloud_image
    wc_b64 = generate_wordcloud_base64(joined) if req.include_wordcloud else ""

    # Also provide a short direct URL using cached image key
    wc_direct_url = None
    wc_key = None
    if req.include_wordcloud and wc_b64:
        try:
            import base64 as _b64
            raw = _b64.b64decode(wc_b64)
            wc_key = cache_wordcloud_image(raw)
            wc_direct_url = str(request.url_for("wordcloud_get_by_key", key=wc_key))
        except Exception:
            wc_direct_url = None
            wc_key = None

    total = len(sentiments)
    def pct(label: str) -> float:
        return round(100.0 * sum(1 for s in sentiments if s.lower() == label) / total, 2) if total else 0.0
    percentages = {
        "positive": pct("positive"),
        "negative": pct("negative"),
        "neutral": pct("neutral"),
    }

    return AnalyzeCompactResponse(
        sentiments=sentiments,
        summary=overall_summary,
        percentages=percentages,
        wordcloud=(WordcloudRef(url=wc_direct_url, key=wc_key) if wc_direct_url and wc_key else None),
    )


@router.post("/compact/upload", response_model=AnalyzeCompactResponse)
async def analyze_compact_upload(
    request: Request,
    file: UploadFile = File(...),
    max_summary_sentences: int = Form(3),
    include_wordcloud: bool = Form(True),
):
    content = await file.read()
    texts = parse_texts_from_upload(file.filename, content)
    if not texts:
        raise HTTPException(status_code=400, detail="No valid texts found in uploaded file")
    # Reuse the same logic as analyze_compact
    try:
        sentiments = analyze_sentiment(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment model error: {e}.")

    joined = " ".join([t for t in texts if t and t.strip()])
    from app.services.summary_service import summarize_text
    overall_summary = summarize_text(joined, max_sentences=max_summary_sentences)

    from app.services.wordcloud_service import generate_wordcloud_base64, cache_wordcloud_image
    wc_b64 = generate_wordcloud_base64(joined) if include_wordcloud else ""
    wc_direct_url = None
    wc_key = None
    if include_wordcloud and wc_b64:
        try:
            import base64 as _b64
            raw = _b64.b64decode(wc_b64)
            wc_key = cache_wordcloud_image(raw)
            wc_direct_url = str(request.url_for("wordcloud_get_by_key", key=wc_key))
        except Exception:
            wc_direct_url = None
            wc_key = None

    total = len(sentiments)
    def pct(label: str) -> float:
        return round(100.0 * sum(1 for s in sentiments if s.lower() == label) / total, 2) if total else 0.0
    percentages = {
        "positive": pct("positive"),
        "negative": pct("negative"),
        "neutral": pct("neutral"),
    }

    return AnalyzeCompactResponse(
        sentiments=sentiments,
        summary=overall_summary,
        percentages=percentages,
        wordcloud=(WordcloudRef(url=wc_direct_url, key=wc_key) if wc_direct_url and wc_key else None),
    )
