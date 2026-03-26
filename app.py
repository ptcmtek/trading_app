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
st.title("Trading Dashboard - Fase 1 + 2")


st.write("Tem key?", "NEWSAPI_KEY" in st.secrets)
if "NEWSAPI_KEY" in st.secrets:
    st.write("Primeiros 5 chars:", st.secrets["NEWSAPI_KEY"][:5])

CONFIG_PATH = Path("config.json")
if not CONFIG_PATH.exists():
    st.error("Falta o ficheiro config.json na mesma pasta do app.py.")
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

SYMBOLS = CONFIG["symbols"]
DEFAULT_SELECTED = CONFIG["default_selected"]
EMA_OPTIONS = CONFIG["ema_options"]
DEFAULT_EMAS = CONFIG["default_emas"]


NEWS_QUERY_MAP = {
    "S&P 500": '"S&P 500" OR SPX OR "US stocks"',
    "Nasdaq 100": '"Nasdaq 100" OR NDX OR Nasdaq OR "US tech stocks"',
    "Dow Jones": '"Dow Jones" OR DJI OR "US industrials"',
    "Russell 2000": '"Russell 2000" OR RUT OR "small cap stocks"',
    "Gold": 'gold OR bullion OR "spot gold" OR XAUUSD',
    "Silver": 'silver OR bullion OR XAGUSD',
    "Oil": 'oil OR crude OR Brent OR WTI',
    "DAX": 'DAX OR "German stocks"',
    "FTSE 100": '"FTSE 100" OR "UK stocks"',
    "Euro Stoxx 50": '"Euro Stoxx 50" OR "euro area stocks"',
}


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
        raise ValueError(f"Não foi encontrada coluna de data. Colunas: {list(df.columns)}")

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
        raise ValueError(
            f"Faltam colunas obrigatórias: {missing}. Colunas disponíveis: {list(df.columns)}"
        )

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

    raise ValueError("Timeframe inválido. Usa '4h' ou '1d'.")


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
            name="Preço",
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
        go.Scatter(x=df["datetime"], y=df["bb_upper"], mode="lines", name="BB Upper"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["datetime"], y=df["bb_mid"], mode="lines", name="BB Mid"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["datetime"], y=df["bb_lower"], mode="lines", name="BB Lower"),
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
        "rebound", "advance", "advances"
    ]
    bearish_terms = [
        "selloff", "drop", "falls", "fall", "lower", "miss", "misses",
        "weak", "bearish", "inflation", "recession", "war", "risk", "crisis",
        "tariff", "volatility", "concern", "slump", "decline"
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_for_label(label: str, page_size: int = 6) -> list[dict]:
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        return [{
            "title": "NEWSAPI_KEY não configurada em st.secrets",
            "sentiment": "Neutral",
            "theme": "Configuração",
            "source": "Sistema",
            "url": "",
            "publishedAt": "",
        }]

    query = NEWS_QUERY_MAP.get(label, label)
    from_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "from": from_date,
    }
    headers = {"X-Api-Key": api_key}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        articles = payload.get("articles", [])
        results = []

        for a in articles:
            title = a.get("title") or "Sem título"
            description = a.get("description") or ""
            source = (a.get("source") or {}).get("name", "")
            published_at = a.get("publishedAt", "")
            article_url = a.get("url", "")

            text = f"{title} {description}".lower()
            sentiment = classify_sentiment_from_text(text)
            theme = classify_theme_from_text(text)

            results.append({
                "title": title,
                "sentiment": sentiment,
                "theme": theme,
                "source": source,
                "url": article_url,
                "publishedAt": published_at,
            })

        return results

    except requests.HTTPError as e:
        try:
            err = resp.json()
            msg = err.get("message", str(e))
        except Exception:
            msg = str(e)

        return [{
            "title": f"Erro News API: {msg}",
            "sentiment": "Neutral",
            "theme": "Erro API",
            "source": "Sistema",
            "url": "",
            "publishedAt": "",
        }]

    except Exception as e:
        return [{
            "title": f"Erro ao obter notícias: {e}",
            "sentiment": "Neutral",
            "theme": "Erro",
            "source": "Sistema",
            "url": "",
            "publishedAt": "",
        }]


def summarize_market_context(news_list):
    if not news_list:
        return {
            "sentiment": "Neutral",
            "themes": "Sem dados",
            "comment": "Sem notícias disponíveis."
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
    themes_text = ", ".join(themes) if themes else "Sem temas"

    if sentiment == "Bearish":
        comment = "Fluxo noticioso com viés negativo para este ativo."
    elif sentiment == "Bullish":
        comment = "Fluxo noticioso com viés positivo para este ativo."
    else:
        comment = "Fluxo noticioso misto, sem direção clara."

    return {
        "sentiment": sentiment,
        "themes": themes_text,
        "comment": comment
    }


def render_news_section(selected_labels):
    st.subheader("Contexto de Mercado (News)")

    for label in selected_labels:
        news = fetch_news_for_label(label)
        context = summarize_market_context(news)

        st.markdown(f"### {label}")

        c1, c2 = st.columns(2)
        c1.metric("Sentimento", context["sentiment"])
        c2.metric("Temas", context["themes"])

        st.markdown(f"**Leitura:** {context['comment']}")

        with st.expander(f"Ver notícias de {label}"):
            if not news:
                st.write("Sem notícias.")
            else:
                for n in news:
                    title = n.get("title", "Sem título")
                    source = n.get("source", "")
                    published = n.get("publishedAt", "")
                    url = n.get("url", "")
                    sentiment = n.get("sentiment", "Neutral")

                    meta = " | ".join(x for x in [source, published, sentiment] if x)

                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

                    if meta:
                        st.caption(meta)


st.sidebar.header("Configuração")

selected = st.sidebar.multiselect(
    "Ativos",
    list(SYMBOLS.keys()),
    default=DEFAULT_SELECTED,
)

timeframe = st.sidebar.selectbox("Timeframe", ["4h", "1d"], index=0)

bars = st.sidebar.slider(
    "Nº de candles",
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
    st.warning("Escolhe pelo menos um ativo.")
    st.stop()

render_news_section(selected)

data_map = {}
summary_rows = []

for label in selected:
    symbol = SYMBOLS[label]

    try:
        df = download_data(symbol, timeframe)

        if df.empty:
            st.warning(f"Sem dados para {label}.")
            continue

        df = df.tail(bars).reset_index(drop=True)
        df = add_indicators(df, emas)

        data_map[label] = df
        sig = compute_signals(df)

        summary_rows.append(
            {
                "Ativo": label,
                "Close": sig["close"],
                "RSI": sig["rsi"],
                "Trend": sig["trend"],
                "Setup": sig["setup"],
                "Estado": sig["state"],
                "Flag": sig["emoji"],
            }
        )

    except Exception as e:
        st.error(f"Erro em {label}: {e}")

st.subheader("Resumo")
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("Sem dados para mostrar.")

for label in selected:
    if label not in data_map:
        continue

    st.subheader(label)
    fig = build_chart(data_map[label], label, emas)
    st.plotly_chart(fig, use_container_width=True)