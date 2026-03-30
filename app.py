import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Trading Dashboard")


CONFIG_PATH = Path("config.json")
if not CONFIG_PATH.exists():
    st.error("Missing config.json in the same folder as app.py.")
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

SYMBOLS = CONFIG["symbols"]
DEFAULT_SELECTED = CONFIG["default_selected"]
EMA_OPTIONS = CONFIG["ema_options"]
DEFAULT_EMAS = CONFIG["default_emas"]


SPECIAL_NEWS_OVERRIDES = {
    "SPX (^GSPC)": {
        "categories": ["general"],
        "keywords": [
            "s&p 500", "sp500", "spx", "^gspc",
            "wall street", "u.s. stocks", "us stocks",
            "equities", "stock market", "index"
        ],
        "priority_keywords": [
            "s&p 500", "spx", "^gspc"
        ],
        "exclude_keywords": [
            "crypto", "bitcoin"
        ],
    },
    "SPX Futures (ES=F)": {
        "categories": ["general"],
        "keywords": [
            "s&p 500 futures", "es futures", "e-mini", "spx futures",
            "s&p 500", "sp500", "equity futures", "wall street futures"
        ],
        "priority_keywords": [
            "s&p 500 futures", "es futures", "e-mini"
        ],
        "exclude_keywords": [
            "crypto", "bitcoin"
        ],
    },
    "XAD6.DE": {
        "categories": ["general"],
        "keywords": [
            "dax", "german stocks", "germany market",
            "frankfurt", "european equities", "euro stocks"
        ],
        "priority_keywords": [
            "dax", "frankfurt", "german stocks"
        ],
        "exclude_keywords": [
            "crypto"
        ],
    },
    "Gold ETC (4GLD.DE)": {
        "categories": ["forex", "general"],
        "keywords": [
            "gold", "bullion", "spot gold", "xauusd",
            "precious metals", "safe haven", "gold etf", "gold etc"
        ],
        "priority_keywords": [
            "gold", "spot gold", "xauusd", "bullion"
        ],
        "exclude_keywords": [],
    },
    "Silver ETC (XAD5.DE)": {
        "categories": ["forex", "general"],
        "keywords": [
            "silver", "spot silver", "xagusd",
            "precious metals", "silver etf", "silver etc", "bullion"
        ],
        "priority_keywords": [
            "silver", "spot silver", "xagusd"
        ],
        "exclude_keywords": [],
    },
    "Gold Futures (GC=F)": {
        "categories": ["forex", "general"],
        "keywords": [
            "gold futures", "comex gold", "gc=f", "gold",
            "spot gold", "xauusd", "bullion", "precious metals"
        ],
        "priority_keywords": [
            "gold futures", "comex gold", "gold", "xauusd"
        ],
        "exclude_keywords": [],
    },
}


def infer_asset_news_profile(label: str, symbol: str) -> dict:
    if label in SPECIAL_NEWS_OVERRIDES:
        return SPECIAL_NEWS_OVERRIDES[label]

    base = label.lower()
    sym = str(symbol).lower()

    categories = ["general"]
    keywords = []
    priority_keywords = []
    exclude_keywords = ["crypto"]

    clean_label = (
        label.replace("(", " ")
        .replace(")", " ")
        .replace("^", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    parts = [p.strip().lower() for p in clean_label.split() if len(p.strip()) > 1]

    keywords.extend(parts[:8])
    priority_keywords.extend(parts[:4])

    if any(x in sym for x in ["usd", "eur", "gbp", "jpy", "xau", "xag", "gc=f", "si=f", "cl=f"]):
        categories = ["forex", "general"]

    if any(x in base for x in ["gold", "silver", "bullion", "xau", "xag"]):
        categories = ["forex", "general"]

    if any(x in base for x in ["oil", "crude", "brent", "wti"]):
        categories = ["general", "forex"]

    if any(x in base for x in ["bond", "treasury", "gilt", "bund"]):
        keywords.extend(["bond yields", "rates", "inflation", "central bank"])
        priority_keywords.extend(["bond yields", "rates"])

    if any(x in base for x in ["world", "msci", "all-world", "acwi"]):
        keywords.extend(["global equities", "world stocks"])
        priority_keywords.extend(["global equities"])

    keywords = list(dict.fromkeys([k for k in keywords if k]))
    priority_keywords = list(dict.fromkeys([k for k in priority_keywords if k]))

    return {
        "categories": categories,
        "keywords": keywords,
        "priority_keywords": priority_keywords,
        "exclude_keywords": exclude_keywords,
    }


def get_asset_news_config(label: str) -> dict:
    symbol = SYMBOLS.get(label, "")
    return infer_asset_news_profile(label, symbol)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []

        for col in df.columns.to_flat_index():
            if isinstance(col, tuple):
                parts = [str(x) for x in col if x is not None]
                matched = None

                for candidate in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                    if candidate in parts:
                        matched = candidate
                        break

                flat_cols.append(matched if matched else "_".join(parts))
            else:
                flat_cols.append(str(col))

        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = df.copy()
    df = flatten_columns(df)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    df = flatten_columns(df)

    datetime_col = None
    for candidate in ["Datetime", "Date", "index", "datetime", "date"]:
        if candidate in df.columns:
            datetime_col = candidate
            break

    if datetime_col is None:
        raise ValueError(f"Date column not found. Columns: {list(df.columns)}")

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    df = df.rename(
        columns={
            datetime_col: "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


@st.cache_data(ttl=900, show_spinner=False)
def download_data(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == "1d":
        raw = yf.download(
            tickers=symbol,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_ohlcv(raw)

    if timeframe == "4h":
        raw = yf.download(
            tickers=symbol,
            period="60d",
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        df = normalize_ohlcv(raw)

        if df.empty:
            return df

        df = df.set_index("datetime")

        df_4h = df.resample("4h", label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        df_4h = df_4h.dropna(subset=["open", "high", "low", "close"]).reset_index()
        df_4h = df_4h.sort_values("datetime").reset_index(drop=True)

        return df_4h

    raise ValueError("Invalid timeframe. Use '4h' or '1d'.")


def add_indicators(df: pd.DataFrame, emas: list[int]) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    for e in emas:
        df[f"ema_{e}"] = close.ewm(span=e, adjust=False).mean()

    df["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def compute_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "close": None,
            "rsi": None,
            "trend": "Neutral",
            "setup": "None",
            "state": "",
            "emoji": "⚠️",
        }

    last = df.iloc[-1]

    trend = "Neutral"
    if "ema_20" in df.columns and "ema_50" in df.columns:
        if pd.notna(last["ema_20"]) and pd.notna(last["ema_50"]):
            if last["close"] < last["ema_20"] < last["ema_50"]:
                trend = "Bearish"
            elif last["close"] > last["ema_20"] > last["ema_50"]:
                trend = "Bullish"

    rsi_state = ""
    if pd.notna(last["rsi"]):
        if last["rsi"] < 30:
            rsi_state = "Oversold"
        elif last["rsi"] > 70:
            rsi_state = "Overbought"

    setup = "None"

    if "ema_20" in df.columns and pd.notna(last.get("ema_20", None)):
        distance_to_ema20 = abs(last["close"] - last["ema_20"]) / last["close"]

        if trend == "Bearish" and distance_to_ema20 < 0.01:
            setup = "Pullback Short"
        elif trend == "Bullish" and distance_to_ema20 < 0.01:
            setup = "Pullback Long"

    if len(df) >= 21:
        prev_20 = df.tail(21).iloc[:-1]
        recent_low = prev_20["low"].min()
        recent_high = prev_20["high"].max()

        if last["close"] < recent_low:
            setup = "Breakdown"
        elif last["close"] > recent_high:
            setup = "Breakout"

    emoji = "✅"
    if rsi_state or setup != "None":
        emoji = "⚠️"

    return {
        "close": round(float(last["close"]), 2),
        "rsi": round(float(last["rsi"]), 1) if pd.notna(last["rsi"]) else None,
        "trend": trend,
        "setup": setup,
        "state": rsi_state,
        "emoji": emoji,
    }


def build_chart(df: pd.DataFrame, label: str, emas: list[int]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    for e in emas:
        col = f"ema_{e}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    mode="lines",
                    name=f"EMA {e}",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["bb_upper"],
            mode="lines",
            name="BB Upper",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["bb_lower"],
            mode="lines",
            name="BB Lower",
            line=dict(width=1),
            fill="tonexty",
            fillcolor="rgba(100, 149, 237, 0.12)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["bb_mid"],
            mode="lines",
            name="BB Mid",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["rsi"],
            mode="lines",
            name="RSI",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_layout(
        height=750,
        title=label,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def classify_sentiment_from_text(text: str) -> str:
    bullish_terms = [
        "rally", "surge", "gain", "gains", "higher", "beats", "beat",
        "strong", "bullish", "optimism", "record high", "upside", "growth",
        "rebound", "advance", "advances",
    ]
    bearish_terms = [
        "selloff", "drop", "falls", "fall", "lower", "miss", "misses",
        "weak", "bearish", "inflation", "recession", "war", "risk", "crisis",
        "tariff", "volatility", "concern", "slump", "decline",
    ]

    bull_score = sum(1 for w in bullish_terms if w in text)
    bear_score = sum(1 for w in bearish_terms if w in text)

    if bull_score > bear_score:
        return "Bullish"
    if bear_score > bull_score:
        return "Bearish"
    return "Neutral"


def classify_theme_from_text(text: str) -> str:
    if any(w in text for w in ["inflation", "cpi", "ppi", "rates", "yield", "fed", "ecb", "interest rate"]):
        return "Rates / Inflation"
    if any(w in text for w in ["earnings", "revenue", "guidance", "results", "quarter", "profit"]):
        return "Earnings"
    if any(w in text for w in ["oil", "crude", "brent", "wti", "gas", "energy"]):
        return "Energy"
    if any(w in text for w in ["war", "tariff", "sanctions", "geopolitical", "middle east", "iran", "china"]):
        return "Geopolitics"
    if any(w in text for w in ["ai", "chip", "semiconductor", "tech", "nvidia", "microsoft", "apple"]):
        return "Technology"
    return "Macro"


def score_news_relevance(label: str, title: str, summary: str) -> int:
    cfg = get_asset_news_config(label)
    keywords = [k.lower() for k in cfg.get("keywords", [])]
    priority_keywords = [k.lower() for k in cfg.get("priority_keywords", [])]
    exclude_keywords = [k.lower() for k in cfg.get("exclude_keywords", [])]

    text = f"{title} {summary}".lower()
    score = 0

    for kw in keywords:
        if kw in text:
            score += 2

    for kw in priority_keywords:
        if kw in text:
            score += 5

    for kw in exclude_keywords:
        if kw in text:
            score -= 4

    macro_terms = ["fed", "ecb", "rates", "inflation", "yield", "earnings", "tariff", "recession"]
    for kw in macro_terms:
        if kw in text:
            score += 1

    return score


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_finnhub_news(category: str, min_date: str) -> list[dict]:
    api_key = st.secrets.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return [{
            "headline": "FINNHUB_API_KEY is not configured in st.secrets",
            "summary": "",
            "source": "System",
            "url": "",
            "datetime": None,
        }]

    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": category,
        "minId": 0,
        "token": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        payload = resp.json()

        if resp.status_code != 200:
            msg = payload.get("error", f"HTTP {resp.status_code}")
            return [{
                "headline": f"Finnhub error: {msg}",
                "summary": "",
                "source": "System",
                "url": "",
                "datetime": None,
            }]

        results = []
        cutoff = datetime.fromisoformat(min_date.replace("Z", "+00:00"))

        for item in payload:
            ts = item.get("datetime")
            dt = None
            if ts:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            if dt and dt < cutoff:
                continue

            results.append({
                "headline": item.get("headline", "No title"),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "datetime": dt,
            })

        return results

    except Exception as e:
        return [{
            "headline": f"Error fetching news: {e}",
            "summary": "",
            "source": "System",
            "url": "",
            "datetime": None,
        }]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_for_label(label: str, page_size: int = 6) -> list[dict]:
    cfg = get_asset_news_config(label)
    categories = cfg.get("categories", ["general"])

    from_date = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ")

    merged = []
    seen = set()

    for category in categories:
        items = fetch_finnhub_news(category, from_date)

        for item in items:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            key = (headline, item.get("url", ""))

            if key in seen:
                continue

            seen.add(key)

            relevance = score_news_relevance(label, headline, summary)
            if relevance < 2:
                continue

            text = f"{headline} {summary}".lower()
            sentiment = classify_sentiment_from_text(text)
            theme = classify_theme_from_text(text)

            merged.append({
                "title": headline,
                "summary": summary,
                "sentiment": sentiment,
                "theme": theme,
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "publishedAt": item.get("datetime").isoformat() if item.get("datetime") else "",
                "relevance": relevance,
            })

    merged = sorted(
        merged,
        key=lambda x: (x.get("relevance", 0), x.get("publishedAt", "")),
        reverse=True
    )

    if not merged:
        return [{
            "title": f"No highly relevant news found for {label}",
            "summary": "",
            "sentiment": "Neutral",
            "theme": "No results",
            "source": "System",
            "url": "",
            "publishedAt": "",
            "relevance": 0,
        }]

    return merged[:page_size]


def summarize_market_context(news_list):
    if not news_list:
        return {
            "sentiment": "Neutral",
            "themes": "No data",
            "comment": "No news available."
        }

    bearish = sum(1 for n in news_list if n["sentiment"] == "Bearish")
    bullish = sum(1 for n in news_list if n["sentiment"] == "Bullish")

    if bearish > bullish:
        sentiment = "Bearish"
    elif bullish > bearish:
        sentiment = "Bullish"
    else:
        sentiment = "Neutral"

    themes = sorted(set(n["theme"] for n in news_list if n.get("theme")))
    themes_text = ", ".join(themes) if themes else "No themes"

    if sentiment == "Bearish":
        comment = "News flow currently leans negative for this asset."
    elif sentiment == "Bullish":
        comment = "News flow currently leans positive for this asset."
    else:
        comment = "News flow is mixed, with no clear directional bias."

    return {
        "sentiment": sentiment,
        "themes": themes_text,
        "comment": comment
    }


def get_lookback_bars(timeframe: str) -> int:
    if timeframe == "1d":
        return 14
    return 14 * 6


def build_technical_analysis(df: pd.DataFrame, label: str, timeframe: str) -> str:
    if df.empty:
        return "No technical data available."

    lookback = get_lookback_bars(timeframe)
    recent = df.tail(lookback).copy()
    if len(recent) < 5:
        return "Not enough recent candles to build a technical view."

    first_close = recent["close"].iloc[0]
    last_close = recent["close"].iloc[-1]
    change_pct = ((last_close / first_close) - 1) * 100 if first_close else 0

    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    trend = "sideways"
    if "ema_20" in recent.columns and "ema_50" in recent.columns:
        ema20 = recent["ema_20"].iloc[-1]
        ema50 = recent["ema_50"].iloc[-1]
        if pd.notna(ema20) and pd.notna(ema50):
            if last_close > ema20 > ema50:
                trend = "bullish"
            elif last_close < ema20 < ema50:
                trend = "bearish"

    momentum_text = "neutral momentum"
    if "rsi" in recent.columns and pd.notna(recent["rsi"].iloc[-1]):
        rsi = recent["rsi"].iloc[-1]
        if rsi > 70:
            momentum_text = f"overbought momentum (RSI {rsi:.1f})"
        elif rsi < 30:
            momentum_text = f"oversold momentum (RSI {rsi:.1f})"
        elif rsi >= 55:
            momentum_text = f"positive momentum (RSI {rsi:.1f})"
        elif rsi <= 45:
            momentum_text = f"weak momentum (RSI {rsi:.1f})"
        else:
            momentum_text = f"balanced momentum (RSI {rsi:.1f})"

    range_position = ""
    if recent_high > recent_low:
        pos = (last_close - recent_low) / (recent_high - recent_low)
        if pos >= 0.8:
            range_position = "Price is trading near the top of its 2-week range."
        elif pos <= 0.2:
            range_position = "Price is trading near the bottom of its 2-week range."
        else:
            range_position = "Price is trading near the middle of its 2-week range."

    breakout_text = ""
    if len(recent) >= 6:
        prev = recent.iloc[:-1]
        prev_high = prev["high"].max()
        prev_low = prev["low"].min()
        if last_close > prev_high:
            breakout_text = "A short-term breakout is in play."
        elif last_close < prev_low:
            breakout_text = "A short-term breakdown is in play."
        else:
            breakout_text = "No confirmed short-term breakout yet."

    direction_sentence = (
        f"Over the last 2 weeks, {label} moved {change_pct:+.2f}% and the structure is currently {trend}."
    )

    return " ".join([
        direction_sentence,
        f"Recent range: {recent_low:.2f} to {recent_high:.2f}.",
        f"Current read: {momentum_text}.",
        range_position,
        breakout_text,
    ]).strip()


def render_news_section(selected_labels):
    st.subheader("Market Context (News)")

    for label in selected_labels:
        news = fetch_news_for_label(label)
        context = summarize_market_context(news)

        st.markdown(f"### {label}")

        c1, c2 = st.columns(2)
        c1.metric("Sentiment", context["sentiment"])
        c2.metric("Themes", context["themes"])

        st.markdown(f"**Read:** {context['comment']}")

        with st.expander(f"View news for {label}"):
            if not news:
                st.write("No news available.")
            else:
                for n in news:
                    title = n.get("title", "No title")
                    summary = n.get("summary", "")
                    source = n.get("source", "")
                    published = n.get("publishedAt", "")
                    url = n.get("url", "")
                    sentiment = n.get("sentiment", "Neutral")
                    relevance = n.get("relevance", 0)

                    meta = " | ".join(
                        x for x in [source, published, sentiment, f"Score {relevance}"] if x
                    )

                    if url:
                        st.markdown(f"**[{title}]({url})**")
                    else:
                        st.markdown(f"**{title}**")

                    if summary:
                        st.write(summary)

                    if meta:
                        st.caption(meta)


st.sidebar.header("Settings")

selected = st.sidebar.multiselect(
    "Assets",
    list(SYMBOLS.keys()),
    default=DEFAULT_SELECTED,
)

timeframe = st.sidebar.selectbox("Timeframe", ["4h", "1d"], index=0)

bars = st.sidebar.slider(
    "Number of candles",
    min_value=50,
    max_value=800,
    value=250,
    step=50,
)

emas = st.sidebar.multiselect(
    "EMAs",
    EMA_OPTIONS,
    default=DEFAULT_EMAS,
)

if not selected:
    st.warning("Select at least one asset.")
    st.stop()

render_news_section(selected)

data_map = {}
summary_rows = {}

for label in selected:
    symbol = SYMBOLS[label]

    try:
        df = download_data(symbol, timeframe)

        if df.empty:
            st.warning(f"No data for {label}.")
            continue

        df = df.tail(bars).reset_index(drop=True)
        df = add_indicators(df, emas)

        data_map[label] = df
        sig = compute_signals(df)

        summary_rows[label] = {
            "Asset": label,
            "Close": sig["close"],
            "RSI": sig["rsi"],
            "Trend": sig["trend"],
            "Setup": sig["setup"],
            "State": sig["state"],
            "Flag": sig["emoji"],
        }

    except Exception as e:
        st.error(f"Error in {label}: {e}")

st.subheader("Summary")
if summary_rows:
    summary_df = pd.DataFrame(list(summary_rows.values()))
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("No data to display.")

for label in selected:
    if label not in data_map:
        continue

    df = data_map[label]

    st.subheader(label)
    technical_text = build_technical_analysis(df, label, timeframe)
    st.markdown("**Technical analysis**")
    st.write(technical_text)

    fig = build_chart(df, label, emas)
    st.plotly_chart(fig, use_container_width=True)