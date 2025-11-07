"""News/Sentiment Agent (simplified and efficient)

This agent strictly fetches news from three allowed sources:
- https://pulse.zerodha.com/
- https://www.nseindia.com/
- https://upstox.com/news/

Features added/changed:
- Robust, lazy-loaded HuggingFace pipeline that prefers Distil/finance models
- Efficient fetching with requests.Session, retries, per-site limits and dedup
- Strong preprocessing: emoji/ticker/url removal and deduplication
- Ticker mapping using `stock_names_symbol.csv` for simple NER-like mapping
- Rolling 5-minute aggregation using pandas
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import re
import time

import requests
from requests.adapters import HTTPAdapter, Retry

import pandas as pd

logger = logging.getLogger(__name__)

# Optional BeautifulSoup
try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

# Lazy singletons
_SENTIMENT_PIPELINE = None
_SENTIMENT_MODEL_NAME = None  # pinned model actually loaded
_TICKER_MAP = None  # Dict[str, List[str]] keys: symbol->aliases (upper)


def _create_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SuperTraderNews/1.0; +https://github.com)"
    })
    return s


def get_default_sources() -> List[str]:
    return [
        "https://pulse.zerodha.com/",
        "https://www.nseindia.com/",
        "https://upstox.com/news/",
    ]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove stock tickers like TICKER, $TICKER or NSE:TICKER
    text = re.sub(r"\$?[A-Z]{2,6}(?:[:._-][A-Z]{1,6})?", " ", text)
    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", " ", text)
    # Remove emojis (common ranges)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF]", flags=re.UNICODE)
    text = emoji_pattern.sub(" ", text)
    # Normalize whitespace and lower
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_ticker_map() -> Dict[str, List[str]]:
    """Load ticker/company mapping from `stock_names_symbol.csv` if available.

    Returns dict: SYMBOL -> [aliases...]
    """
    global _TICKER_MAP
    if _TICKER_MAP is not None:
        return _TICKER_MAP
    path = "stock_names_symbol.csv"
    try:
        df = pd.read_csv(path)
        # Expect columns like 'symbol' and 'name' (be tolerant)
        sym_col = None
        name_col = None
        for c in df.columns:
            if c.lower() in ("symbol", "ticker"):
                sym_col = c
            if c.lower() in ("name", "company"):
                name_col = c
        mapping: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            sym = str(row[sym_col]) if sym_col else None
            name = str(row[name_col]) if name_col else None
            if not sym and not name:
                continue
            key = (sym or name).upper()
            aliases = [a.upper() for a in {sym, name} if a and str(a).strip()]
            mapping[key] = aliases
        _TICKER_MAP = mapping
        return _TICKER_MAP
    except Exception:
        _TICKER_MAP = {}
        return _TICKER_MAP


def _simple_map_to_tickers(text: str) -> List[str]:
    """Simple mapping: find known symbols or company names appearing in text."""
    mapping = _load_ticker_map()
    if not mapping:
        return []
    txt = text.upper()
    found = set()
    # simple substring match for aliases
    for key, aliases in mapping.items():
        for a in aliases:
            if not a:
                continue
            if a in txt:
                found.add(key)
    return list(found)


def fetch_news_from_sources(sources: Optional[List[str]] = None, max_items_per_site: int = 6) -> List[Dict[str, Any]]:
    """Fetch and return cleaned, deduplicated news articles from the allowed sources.

    Each returned item contains: source, url, title, text, cleaned_text, timestamp
    """
    sess = _create_session()
    sources = sources or get_default_sources()
    items: List[Dict[str, Any]] = []
    seen_hashes = set()

    for site in sources:
        try:
            resp = sess.get(site, timeout=8)
            resp.raise_for_status()
            html = resp.text
            snippets = _parse_html_for_texts(html, base_url=site, session=sess, max_links=max_items_per_site)
            for sn in snippets[:max_items_per_site]:
                title = sn.get("title") or ""
                summary = sn.get("summary") or ""
                url = sn.get("url") or site
                raw_text = f"{title} {summary}".strip()
                cleaned = _clean_text(raw_text)
                if not cleaned:
                    continue
                h = hash(cleaned[:300])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                items.append({
                    "source": site,
                    "url": url,
                    "title": title,
                    "text": raw_text,
                    "cleaned_text": cleaned,
                    "timestamp": datetime.utcnow(),
                })
        except Exception as e:
            logger.debug(f"Failed to fetch {site}: {e}")

    return items


def _parse_html_for_texts(html: str, base_url: str, session: requests.Session, max_links: int = 6) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if BeautifulSoup is None:
        # fallback minimal parser
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if m:
            results.append({"title": m.group(1).strip(), "summary": "", "url": base_url})
        return results

    soup = BeautifulSoup(html, "html.parser")
    # Title and meta
    if soup.title and soup.title.string:
        results.append({"title": soup.title.string.strip(), "summary": "", "url": base_url})
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if meta and meta.get("content"):
        results.append({"title": (meta.get("content") or "")[:140].strip(), "summary": meta.get("content"), "url": base_url})

    # Find prominent article/heading links
    count = 0
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True)
        href = a.get("href")
        if not txt or len(txt) < 30:
            continue
        # normalize url
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href.startswith("http") and base_url.split("//")[1].split("/")[0] in href:
            results.append({"title": txt, "summary": "", "url": href})
            count += 1
        if count >= max_links:
            break

    return results


DEFAULT_MODEL_CANDIDATES = [
    "yashkumar/distil-finbert",  # distil-size finbert (community)
    "ProsusAI/finbert",          # original FinBERT
    "yiyanghkust/finbert-tone",  # alt FinBERT tone
    "cardiffnlp/twitter-roberta-base-sentiment",  # general sentiment
    "distilbert-base-uncased-finetuned-sst-2-english",  # light baseline
]


def configure_sentiment_model(model_name: Optional[str] = None, force_reload: bool = False) -> Optional[str]:
    """Explicitly configure (and load) a HuggingFace sentiment model.

    Args:
        model_name: Specific model to load. If None, will consult env var NEWS_SENTIMENT_MODEL, then fall back to defaults.
        force_reload: If True, will drop any existing pipeline and reload.

    Returns:
        The loaded model name or None if fallback (lexicon) will be used.
    """
    global _SENTIMENT_PIPELINE, _SENTIMENT_MODEL_NAME
    if _SENTIMENT_PIPELINE is not None and not force_reload:
        return _SENTIMENT_MODEL_NAME

    # Reset if forcing
    if force_reload:
        _SENTIMENT_PIPELINE = None
        _SENTIMENT_MODEL_NAME = None

    # Determine desired model
    desired = model_name or os.getenv("NEWS_SENTIMENT_MODEL")
    candidates = []
    if desired:
        candidates.append(desired)
    candidates.extend([m for m in DEFAULT_MODEL_CANDIDATES if m != desired])

    try:
        from transformers import pipeline  # noqa: F401
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()  # reduce noise
    except Exception as e:
        logger.debug(f"transformers unavailable for sentiment model: {e}")
        return None

    for m in candidates:
        try:
            _SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model=m, truncation=True)
            _SENTIMENT_MODEL_NAME = m
            logger.info(f"Sentiment model configured: {m}")
            return m
        except Exception as e:
            logger.debug(f"Failed loading {m}: {e}")

    logger.warning("All candidate models failed; using lexicon fallback.")
    return None


def _ensure_sentiment_pipeline(prefer: Optional[str] = None):
    """Lazy load sentiment pipeline using configure_sentiment_model if not yet loaded."""
    if _SENTIMENT_PIPELINE is not None:
        return
    configure_sentiment_model(prefer_model_or_env(prefer))


def prefer_model_or_env(explicit: Optional[str]) -> Optional[str]:
    """Resolve preferred model via explicit argument or environment variable."""
    return explicit or os.getenv("NEWS_SENTIMENT_MODEL")


def score_sentiment_batch(texts: List[str], model_name: Optional[str] = None) -> List[float]:
    """Score a batch of texts and return sentiment in [-1, 1].

    Uses DistilFinBERT-like models when available, otherwise a fast lexicon fallback.
    """
    texts_clean = [t.strip() for t in texts]
    # Allow caller to force a model (first call) by passing model_name
    if model_name and (_SENTIMENT_PIPELINE is None or _SENTIMENT_MODEL_NAME != model_name):
        configure_sentiment_model(model_name, force_reload=True)
    _ensure_sentiment_pipeline(model_name)
    if _SENTIMENT_PIPELINE is not None:
        try:
            outs = _SENTIMENT_PIPELINE(texts_clean, truncation=True)
            scores: List[float] = []
            for out in outs:
                lab = str(out.get("label", "")).lower()
                sc = float(out.get("score", 0.0))
                if "neg" in lab or "negative" in lab:
                    scores.append(-sc)
                elif "pos" in lab or "positive" in lab:
                    scores.append(sc)
                elif "neutral" in lab:
                    scores.append(0.0)
                else:
                    # Unknown label mapping
                    scores.append((sc if "pos" in lab else 0.0))
            return scores
        except Exception as e:
            logger.debug(f"Transformer pipeline failed at scoring: {e}")

    # Fallback lexicon-based approach (fast)
    POS = {"gain", "up", "rise", "bull", "positive", "beat", "surge", "profit", "rally", "good", "buy", "gain"}
    NEG = {"loss", "down", "fall", "drop", "bear", "negative", "miss", "crash", "weak", "sell", "decline"}
    results = []
    for t in texts_clean:
        if not t:
            results.append(0.0)
            continue
        words = set(re.findall(r"\w+", t.lower()))
        p = len(words & POS)
        n = len(words & NEG)
        if p == n:
            results.append(0.0)
        else:
            v = (p - n) / max(p + n, 1)
            results.append(max(-1.0, min(1.0, v)))
    return results


def map_news_to_tickers(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach detected tickers to news items using a simple mapping."""
    for it in news_items:
        txt = it.get("cleaned_text") or it.get("text") or ""
        it["tickers"] = _simple_map_to_tickers(txt)
    return news_items


def aggregate_rolling_sentiment(news_items: List[Dict[str, Any]], window_minutes: int = 5) -> pd.DataFrame:
    """Compute rolling mean sentiment over a time-indexed DataFrame (window in minutes).

    Returns a DataFrame indexed by timestamp with columns: sentiment_mean, count
    """
    if not news_items:
        return pd.DataFrame(columns=["sentiment_mean", "count"]) 
    df = pd.DataFrame([
        {"ts": (it.get("timestamp") if isinstance(it.get("timestamp"), datetime) else datetime.utcnow()),
         "text": it.get("cleaned_text") or it.get("text") or "",
         "source": it.get("source")}
        for it in news_items
    ])
    if df.empty:
        return pd.DataFrame(columns=["sentiment_mean", "count"]) 
    df["ts"] = pd.to_datetime(df["ts"])
    # Score in batch
    df["score"] = score_sentiment_batch(df["text"].tolist())
    df = df.sort_values("ts")
    df = df.set_index("ts")
    # Resample into 1-minute bins first
    r = df["score"].resample("1T").mean().fillna(0)
    roll = r.rolling(f"{window_minutes}T").mean()
    out = pd.DataFrame({"sentiment_mean": roll, "count": df["score"].resample("1T").count()})
    return out


def get_market_sentiment_live() -> Dict[str, Any]:
    """Top-level helper: fetch, map tickers, score, and return 5-min aggregated snapshot."""
    items = fetch_news_from_sources(get_default_sources(), max_items_per_site=6)
    items = map_news_to_tickers(items)
    agg = aggregate_rolling_sentiment(items, window_minutes=5)
    if agg.empty:
        return {
            "market_sentiment_5min": 0.0,
            "sentiment_timeseries": [],
            "high_impact_news": [],
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.0,
        }
    latest = agg.dropna().iloc[-1]
    # pick top 3 long texts as high impact heuristics
    high_impact = sorted(items, key=lambda x: len(x.get("cleaned_text", "")), reverse=True)[:3]
    return {
        "market_sentiment_5min": float(latest.get("sentiment_mean", 0.0)),
        "sentiment_timeseries": agg["sentiment_mean"].fillna(0).tail(12).tolist(),
        "high_impact_news": high_impact,
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": 0.7 if _SENTIMENT_PIPELINE is not None else 0.35,
    }


__all__ = [
    "get_default_sources",
    "fetch_news_from_sources",
    "score_sentiment_batch",
    "map_news_to_tickers",
    "aggregate_rolling_sentiment",
    "get_market_sentiment_live",
    "configure_sentiment_model",
]