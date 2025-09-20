import os
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# Redirect caches away from C: to a local project folder on E:
try:
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _cache_root = os.path.join(_base_dir, ".cache")
    os.makedirs(_cache_root, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache_root, "hf"))
    os.environ.setdefault("HF_HOME", os.path.join(_cache_root, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_cache_root, "hf", "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(_cache_root, "torch"))
except Exception:
    # If setting cache dirs fails, continue with defaults
    pass

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig

MODEL_NAME = 'ProsusAI/finbert'
BATCH_SIZE = 16
MAX_LENGTH = 128
ENSEMBLE_WEIGHT = 0.7

label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
inverse_label_map = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text = str(self.texts[index])
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten()
        }

@torch.no_grad()
def get_transformer_probs(model, data_loader, device):
    model.eval()
    probs = []
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
    return np.array(probs)

class HybridSentimentPredictor:
    def __init__(self, finbert_path=None, svm_path=None, tfidf_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        finbert_path = finbert_path or os.path.join(base_dir, 'FINBERT_FINAL.BIN')
        svm_path = svm_path or os.path.join(base_dir, 'SVM_FINAL.PKL')
        tfidf_path = tfidf_path or os.path.join(base_dir, 'TFIDF_VECTORIZER_FINAL.PKL')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        config = BertConfig(num_labels=len(label_map))
        self.finbert_model = AutoModelForSequenceClassification.from_config(config)
        sd = torch.load(finbert_path, map_location=device)
        self.finbert_model.load_state_dict(sd)
        self.finbert_model.to(device).eval()

        self.svm_model = None
        self.tfidf_vectorizer = None
        if os.path.exists(svm_path) and os.path.exists(tfidf_path):
            try:
                self.svm_model = joblib.load(svm_path)
                self.tfidf_vectorizer = joblib.load(tfidf_path)
            except Exception:
                self.svm_model = None
                self.tfidf_vectorizer = None

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts or all(not str(t).strip() for t in texts):
            return ["neutral"] * max(len(texts), 1)

        dataset = SentimentDataset(texts, self.tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(texts)))
        finbert_probs = get_transformer_probs(self.finbert_model, loader, device)

        if self.svm_model is not None and self.tfidf_vectorizer is not None:
            tfidf = self.tfidf_vectorizer.transform(texts)
            decision = self.svm_model.decision_function(tfidf)
            if decision.ndim == 1:
                decision = decision.reshape(1, -1)
            expd = np.exp(decision - decision.max(axis=1, keepdims=True))
            svm_probs = expd / expd.sum(axis=1, keepdims=True)
            combined = (ENSEMBLE_WEIGHT * finbert_probs) + ((1 - ENSEMBLE_WEIGHT) * svm_probs)
        else:
            combined = finbert_probs

        idx = np.argmax(combined, axis=1)
        return [inverse_label_map[i] for i in idx]

def main():
    parser = argparse.ArgumentParser(description='Hybrid Sentiment Predictor')
    parser.add_argument('--finbert', dest='finbert_path', help='Path to FINBERT_FINAL.BIN')
    parser.add_argument('--svm', dest='svm_path', help='Path to SVM_FINAL.PKL')
    parser.add_argument('--tfidf', dest='tfidf_path', help='Path to TFIDF_VECTORIZER_FINAL.PKL')
    parser.add_argument('--text', help='Single text to analyze')
    parser.add_argument('--interactive', action='store_true', help='Run interactive input loop')
    args = parser.parse_args()

    predictor = HybridSentimentPredictor(args.finbert_path, args.svm_path, args.tfidf_path)

    if args.text:
        print(predictor.predict(args.text)[0])
        return

    if args.interactive:
        print("Enter a comment (type 'exit' to quit):")
        while True:
            user_input = input('> ')
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                print('Please enter some text to analyze.')
                continue
            print(predictor.predict(user_input)[0])
        return

    # Demo
    demo = [
        "The recent amendments regarding startup compliance will definitely boost investor confidence.",
        "While we appreciate the intent to simplify, the new compliance rules create a significant burden for MSMEs.",
        "This is a major oversight that undermines the entire effort to help listed companies.",
        "Can the ministry provide clarification on how the new ESG reporting rules will impact auditors?",
        "This is a fantastic feature, I absolutely love it!",
        "The service was terrible and I am very disappointed.",
        "What a horrible experience, I would not recommend this to anyone.",
        "Okay, I understand. Thank you for the information.",
        "what the helly",
    ]
    preds = predictor.predict(demo)
    for t, s in zip(demo, preds):
        print(f'Text: "{t}"\n  -> Predicted Sentiment: {s.upper()}\n')

if __name__ == '__main__':
    main()
