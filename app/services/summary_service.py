from typing import List

# Simple extractive summarization baseline: pick top-N sentences by length/heuristic.
# For production, replace with a transformer summarizer (e.g., facebook/bart-large-cnn)
# or pegasus via Hugging Face. This placeholder avoids heavy downloads.

def summarize_text(text: str, max_sentences: int = 3) -> str:
    if not text or not text.strip():
        return ""
    # Split on sentence boundaries (very naive)
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    # Rank sentences by length (proxy for information)
    sentences_sorted = sorted(sentences, key=lambda s: len(s), reverse=True)
    top = sentences_sorted[:max_sentences]
    # Preserve original order
    order = {s: i for i, s in enumerate(sentences)}
    top_sorted = sorted(top, key=lambda s: order.get(s, 0))
    return " ".join(top_sorted)

def summarize_batch(texts: List[str], max_sentences: int = 3) -> List[str]:
    return [summarize_text(t, max_sentences=max_sentences) for t in texts]
