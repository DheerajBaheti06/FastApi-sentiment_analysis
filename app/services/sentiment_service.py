import os
from typing import List, Optional

_predictor_instance = None

def _default_path_if_exists(*paths: str) -> Optional[str]:
    p = os.path.join(*paths)
    return p if os.path.exists(p) else None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        # Lazy import to avoid requiring heavy deps until first use
        from predictor import HybridSentimentPredictor
        # Prefer env vars
        finbert_path = os.getenv('FINBERT_PATH')
        svm_path = os.getenv('SVM_PATH')
        tfidf_path = os.getenv('TFIDF_PATH')

        # Autodiscover under app/models if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
        models_dir = os.path.join(base_dir, 'models')
        finbert_path = finbert_path or _default_path_if_exists(models_dir, 'FINBERT_FINAL.BIN')
        svm_path = svm_path or _default_path_if_exists(models_dir, 'SVM_FINAL.PKL')
        tfidf_path = tfidf_path or _default_path_if_exists(models_dir, 'TFIDF_VECTORIZER_FINAL.PKL')

        _predictor_instance = HybridSentimentPredictor(finbert_path, svm_path, tfidf_path)
    return _predictor_instance

def analyze_sentiment(texts: List[str]) -> List[str]:
    predictor = get_predictor()
    return predictor.predict(texts)
