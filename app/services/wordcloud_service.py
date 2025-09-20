import base64
import io
import re
from typing import List, Optional, Tuple
import time
import secrets

try:
    from wordcloud import WordCloud, STOPWORDS
except Exception:
    WordCloud = None
    STOPWORDS = set()


def _encode_jpeg_base64(pil_image, quality: int = 70) -> str:
    buf = io.BytesIO()
    # Ensure RGB for JPEG
    img = pil_image.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_wordcloud_base64(
    text: str,
    width: int = 480,
    height: int = 240,
    max_words: int = 80,
    size_cap: int = 200_000,  # max base64 string length
) -> str:
    if not text or not text.strip() or WordCloud is None:
        return ""
    # If there are no word characters, wordcloud will raise; skip gracefully
    if not re.search(r"\w", text):
        return ""
    try:
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            stopwords=STOPWORDS,
            collocations=False,
            max_words=max_words,
        )
        img = wc.generate(text).to_image()
    except Exception:
        # Includes ValueError for no words after internal preprocessing
        return ""

    # Try a few encode settings to keep payload reasonable
    for q in (70, 60, 50):
        b64 = _encode_jpeg_base64(img, quality=q)
        if len(b64) <= size_cap:
            return b64
    # As a fallback, downscale and try again
    try:
        small = img.resize((max(240, width // 2), max(120, height // 2)))
        b64 = _encode_jpeg_base64(small, quality=60)
        return b64
    except Exception:
        return ""


def generate_wordclouds(texts: List[str], width: int = 800, height: int = 400) -> List[str]:
    return [generate_wordcloud_base64(t, width=width, height=height) for t in texts]


# -- Simple in-memory cache for generated wordcloud images --
_WC_CACHE: dict[str, Tuple[bytes, float]] = {}
_WC_TTL_SECONDS = 10 * 60  # 10 minutes
_WC_MAX_ITEMS = 200


def _cleanup_cache() -> None:
    now = time.time()
    expired = [k for k, (_, ts) in _WC_CACHE.items() if (now - ts) > _WC_TTL_SECONDS]
    for k in expired:
        _WC_CACHE.pop(k, None)
    # Trim if oversized
    if len(_WC_CACHE) > _WC_MAX_ITEMS:
        # Drop oldest items
        items = sorted(_WC_CACHE.items(), key=lambda it: it[1][1])
        to_drop = len(_WC_CACHE) - _WC_MAX_ITEMS
        for k, _ in items[:to_drop]:
            _WC_CACHE.pop(k, None)


def cache_wordcloud_image(image_bytes: bytes) -> str:
    _cleanup_cache()
    key = secrets.token_urlsafe(8)
    _WC_CACHE[key] = (image_bytes, time.time())
    return key


def get_cached_wordcloud_image(key: str) -> Optional[bytes]:
    item = _WC_CACHE.get(key)
    if not item:
        return None
    img, ts = item
    if (time.time() - ts) > _WC_TTL_SECONDS:
        _WC_CACHE.pop(key, None)
        return None
    return img
