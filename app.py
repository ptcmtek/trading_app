import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Trading Dashboard - Fase 1 + 2")

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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(str(col[0]))
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
        raise ValueError(f"Faltam colunas obrigatórias: {missing}. Colunas: {list(df.columns)}")

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
            symbol,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_ohlcv(raw)

    if timeframe == "4h":
        raw = yf.download(
            symbol,
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

        df_4h = df.resample("4h").agg(
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
    last = df.iloc[-1]

    trend = "Neutral"
    if "ema_20" in df.columns and "ema_50" in df.columns:
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

    if "ema_20" in df.columns and pd.notna(last["ema_20"]):
        distance_to_ema20 = abs(last["close"] - last["ema_20"]) / last["close"]

        if trend == "Bearish" and distance_to_ema20 < 0.01:
            setup = "Pullback Short"

        if trend == "Bullish" and distance_to_ema20 < 0.01:
            setup = "Pullback Long"

    prev_20 = df.tail(21).iloc[:-1]
    if not prev_20.empty:
        recent_low = prev_20["low"].min()
        recent_high = prev_20["high"].max()

        if last["close"] < recent_low:
            setup = "Breakdown"

        if last["close"] > recent_high:
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

# ---------------- NEWS (SIMPLE MOCK + STRUCTURE) ----------------
# Fase inicial: estrutura pronta para depois ligar API real

def get_mock_news():
    return [
        {"title": "Oil prices rising amid supply concerns", "sentiment": "Bearish", "theme": "Energy / Inflation"},
        {"title": "US yields remain elevated", "sentiment": "Bearish", "theme": "Rates"},
        {"title": "Tech earnings mixed outlook", "sentiment": "Neutral", "theme": "Earnings"},
    ]


def summarize_market_context(news_list):
    bearish = sum(1 for n in news_list if n["sentiment"] == "Bearish")
    bullish = sum(1 for n in news_list if n["sentiment"] == "Bullish")

    if bearish > bullish:
        sentiment = "Bearish"
    elif bullish > bearish:
        sentiment = "Bullish"
    else:
        sentiment = "Neutral"

    themes = list(set(n["theme"] for n in news_list))

    comment = ""
    if sentiment == "Bearish":
        comment = "Fluxo noticioso com viés negativo para equities."
    elif sentiment == "Bullish":
        comment = "Fluxo noticioso com suporte ao risco."
    else:
        comment = "Fluxo noticioso misto, sem direção clara."

    return {
        "sentiment": sentiment,
        "themes": ", ".join(themes),
        "comment": comment
    }


def render_news_section():
    st.subheader("Contexto de Mercado (News)")

    news = get_mock_news()
    context = summarize_market_context(news)

    c1, c2 = st.columns(2)
    c1.metric("Sentimento", context["sentiment"])
    c2.metric("Temas", context["themes"])

    st.markdown(f"**Leitura:** {context['comment']}")

    with st.expander("Ver notícias"):
        for n in news:
            st.markdown(f"- {n['title']} ({n['sentiment']})")


# ---------------- RENDER NEWS BEFORE DATA ----------------
render_news_section()

