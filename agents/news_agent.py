import logging
import os
from typing import Any, Dict, Optional

import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

NER_MODEL = None
SENTIMENT_PIPE = None
ALIAS_DICT: Dict[str, str] = {}  # e.g., {"Tata Power": "NSE:TATAPOWER", "Adani Green": "NSE:ADANIGREEN"}

def init_news_agent(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the News/Sentiment Agent by loading NER and sentiment models.
    """

    global NER_MODEL, SENTIMENT_PIPE, ALIAS_DICT

    config = config or {}
    ner_model_name = config.get("ner_model", "en_core_web_trf")
    sentiment_model_name = config.get("sentiment_model", "ProsusAI/finbert")
    alias_path = config.get("alias_dict")

    try:
        logger.info("Loading NER model: %s", ner_model_name)
        NER_MODEL = spacy.load(ner_model_name)
        logger.info("NER model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load NER model")
        raise RuntimeError(f"NER load error: {e}")

    try:
        logger.info("Loading sentiment model: %s", sentiment_model_name)
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        SENTIMENT_PIPE = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info("Sentiment pipeline ready")
    except Exception as e:
        logger.exception("Failed to load sentiment model")
        raise RuntimeError(f"Sentiment model load error: {e}")

    if alias_path and os.path.exists(alias_path):
        import json

        with open(alias_path, "r") as f:
            ALIAS_DICT = json.load(f)
        logger.info("Alias dictionary loaded with %d entries", len(ALIAS_DICT))
    else:
        ALIAS_DICT = {}
        logger.warning("No alias_dict provided, entity linking may be incomplete")

    ctx = {
        "ner_model": ner_model_name,
        "sentiment_model": sentiment_model_name,
        "alias_dict_size": len(ALIAS_DICT),
    }
    return ctx

