from fastapi import APIRouter, Query, Response, HTTPException
from fastapi import Request
from pydantic import BaseModel
from app.services.wordcloud_service import (
    generate_wordcloud_base64,
    cache_wordcloud_image,
    get_cached_wordcloud_image,
)
import base64

router = APIRouter(prefix="/wordcloud", tags=["wordcloud"]) 

@router.get("")
def wordcloud(text: str = Query(..., description="Text to render as word cloud")):
    b64 = generate_wordcloud_base64(text)
    if not b64:
        # Return empty 204 when cannot generate (no words)
        return Response(status_code=204)
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return Response(status_code=500)
    return Response(content=raw, media_type="image/jpeg")


class WordcloudBody(BaseModel):
    texts: list[str]


@router.post("/generate")
def wordcloud_generate(body: WordcloudBody, request: Request):
    joined = " ".join([t for t in body.texts if t and t.strip()])
    b64 = generate_wordcloud_base64(joined)
    if not b64:
        raise HTTPException(status_code=422, detail="No valid words to generate wordcloud")
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to decode image")
    key = cache_wordcloud_image(raw)
    # Build absolute URL for the cached image
    url = str(request.url_for("wordcloud_get_by_key", key=key))
    return {"url": url, "key": key}


@router.get("/{key}", name="wordcloud_get_by_key")
def wordcloud_get_by_key(key: str):
    raw = get_cached_wordcloud_image(key)
    if raw is None:
        raise HTTPException(status_code=404, detail="Not found or expired")
    return Response(content=raw, media_type="image/jpeg")
