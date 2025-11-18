# Loads trained model, runs preprocessing and prediction

import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DISTILBERT_LOCAL_DIR, MAX_LEN_BERT, CONF_THRESHOLD

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model_once():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_LOCAL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_LOCAL_DIR))
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model

def predict(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    txt = text.strip()
    if not txt:
        return {"label": "n/a", "prob": 0.0, "latency_ms": 0, "device": str(_device), "error": "Empty input."}

    tok, mdl = _load_model_once()

    # Belt-and-suspenders: ensure model is on the same device you’re using for inputs
    if next(mdl.parameters()).device.type != _device.type:
        mdl.to(_device)

    enc = tok(txt, truncation=True, max_length=MAX_LEN_BERT, return_tensors="pt")
    for k in enc.keys():
        enc[k] = enc[k].to(_device)

    start = time.time()
    with torch.no_grad():
        out = mdl(**enc)
        logits = out.logits.detach().cpu().numpy()[0]
        probs = softmax(logits)
    elapsed = (time.time() - start) * 1000.0

    prob_pos = float(probs[1])
    label = "pos" if prob_pos >= CONF_THRESHOLD else "neg"
    return {"label": label, "prob": prob_pos, "latency_ms": round(elapsed, 2), "device": str(_device)}

def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
