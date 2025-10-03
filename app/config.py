# Stores configuration

from pathlib import Path

# Resolve the repo root (…/customer-review-analyzer)
BASE_DIR = Path(__file__).resolve().parents[1]

# Absolute path to your fine-tuned DistilBERT folder
DISTILBERT_LOCAL_DIR = BASE_DIR / "models" / "distilbert_imdb"

CONF_THRESHOLD = 0.50 # decision threshold for positive
MAX_LEN_BERT = 256 # what I trained with
APP_TITLE = "IMDB Review Sentiment Analyzer"
APP_TAGLINE = "Paste a movie review and get a sentiment prediction using a fine-tuned DistilBERT."