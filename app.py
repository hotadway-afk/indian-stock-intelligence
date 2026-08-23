import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================
# Indian Stock Intelligence — V2.1
# Fundamental + Quarterly + Valuation + Technical + Risk Engine
# Data source: Yahoo Finance via yfinance
# ============================================================

st.set_page_config(
    page_title="Indian Stock Intelligence V2.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------

def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def money(x):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"₹{x/1e12:.2f}T"
    if ax >= 1e9:
        return f"₹{x/1e9:.2f}B"
    if ax >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if ax >= 1e5:
        return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"


def latest_value(df, row_names):
    if df is None or df.empty:
        return np.nan
    for row in row_names:
        if row in df.index:
            s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
            if not s.empty:
                return float(s.iloc[0])
    return np.nan


def previous_value(df, row_names):
    if df is None or df.empty:
        return np.nan
    for row in row_names:
        if row in df.index:
            s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
            if len(s) >= 2:
                return float(s.iloc[1])
    return np.nan


def growth(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return current / previous - 1


def normalize_series(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.empty or s.max() == s.min():
        return pd.Series(50.0, index=s.index)
    return 100 * (s - s.min()) / (s.max() - s.min())


def score_higher_better(value, good, excellent):
    if pd.isna(value):
        return 50.0
    if value >= excellent:
        return 100.0
    if value <= 0:
        return 0.0
    return max(0.0, min(100.0, 100 * (value / excellent)))


def score_lower_better(value, bad, good):
    if pd.isna(value):
        return 50.0
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return max(0.0, min(100.0, 100 * (bad - value) / (bad - good)))


def safe_concat_frames(frames):
    frames = [x for x in frames if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0)


# -----------------------------
# Technical indicators
# -----------------------------

def add_indicators(df):
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    for n in [20, 50, 100, 200]:
        out[f"SMA{n}"] = close.rolling(n).mean()

    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["ATR_pct"] = out["ATR14"] / close

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).sum() / atr14.rolling(14).sum()
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).sum() / atr14.rolling(14).sum()
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["ADX14"] = dx.rolling(14).mean()

    out["Vol20"] = volume.rolling(20).mean()
    out["VolRatio"] = volume / out["Vol20"]
    out["52W_High"] = close.rolling(252, min_periods=20).max()
    out["52W_Low"] = close.rolling(252, min_periods=20).min()

    out["Return20"] = close / close.shift(20) - 1
    out["Return60"] = close / close.shift(60) - 1
    out["Return120"] = close / close.shift(120) - 1

    return out


def technical_score(d):
    if d.empty:
        return 50.0, []

    r = d.iloc[-1]
    score = 50.0
    positives, negatives = [], []

    close = safe_float(r["Close"])
    sma20 = safe_float(r["SMA20"])
    sma50 = safe_float(r["SMA50"])
    sma200 = safe_float(r["SMA200"])
    rsi = safe_float(r["RSI14"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_signal"])
    adx = safe_float(r["ADX14"])
    vol_ratio = safe_float(r["VolRatio"])

    if not pd.isna(sma50) and close > sma50:
        score += 10
        positives.append("Price is above 50-DMA.")
    elif not pd.isna(sma50):
        score -= 10
        negatives.append("Price is below 50-DMA.")

    if not pd.isna(sma200) and close > sma200:
        score += 12
        positives.append("Price is above 200-DMA.")
    elif not pd.isna(sma200):
        score -= 12
        negatives.append("Price is below 200-DMA.")

    if not pd.isna(sma50) and not pd.isna(sma200) and sma50 > sma200:
        score += 8
        positives.append("50-DMA is above 200-DMA.")
    elif not pd.isna(sma50) and not pd.isna(sma200):
        score -= 8
        negatives.append("50-DMA is below 200-DMA.")

    if not pd.isna(rsi):
        if 50 <= rsi <= 68:
            score += 8
            positives.append(f"RSI is constructive at {rsi:.1f}.")
        elif rsi > 75:
            score -= 5
            negatives.append(f"RSI is overbought at {rsi:.1f}.")
        elif rsi < 35:
            score -= 8
            negatives.append(f"RSI is weak at {rsi:.1f}.")

    if not pd.isna(macd) and not pd.isna(signal):
        if macd > signal:
            score += 7
            positives.append("MACD is above its signal line.")
        else:
            score -= 7
            negatives.append("MACD is below its signal line.")

    if not pd.isna(adx):
        if adx >= 25:
            score += 5
            positives.append(f"ADX {adx:.1f} indicates a meaningful trend.")
        elif adx < 15:
            negatives.append(f"ADX {adx:.1f} indicates a weak trend.")

    if not pd.isna(vol_ratio) and vol_ratio >= 1.5:
        score += 5
        positives.append("Volume is elevated versus its 20-day average.")

    return max(0, min(100, score)), positives + negatives


# -----------------------------
# Data download
# -----------------------------

@st.cache_data(ttl=900, show_spinner=False)
def load_stock(symbol):
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period="5y", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data returned for {symbol}.")

    hist = hist.reset_index()
    hist.columns = [str(c).replace(" ", "_") for c in hist.columns]
    if "Datetime" in hist.columns and "Date" not in hist.columns:
        hist = hist.rename(columns={"Datetime": "Date"})

    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist = hist.set_index("Date")
    hist = hist[["Open", "High", "Low", "Close", "Adj_Close", "Volume"]] if "Adj_Close" in hist.columns else hist[["Open", "High", "Low", "Close", "Volume"]]
    hist = hist.dropna(subset=["Close"])

    # yfinance financial statements are DataFrames and are safe to cache.
    annual_income = ticker.financials
    annual_balance = ticker.balance_sheet
    annual_cashflow = ticker.cashflow
    quarterly_income = ticker.quarterly_financials
    quarterly_balance = ticker.quarterly_balance_sheet
    quarterly_cashflow = ticker.quarterly_cashflow

    try:
        info = ticker.info
    except Exception:
        info = {}

    try:
        news = ticker.news
    except Exception:
        news = []

    return {
        "hist": hist,
        "annual_income": annual_income,
        "annual_balance": annual_balance,
        "annual_cashflow": annual_cashflow,
        "quarterly_income": quarterly_income,
        "quarterly_balance": quarterly_balance,
        "quarterly_cashflow": quarterly_cashflow,
        "info": info if isinstance(info, dict) else {},
        "news": news if isinstance(news, list) else [],
    }


# -----------------------------
# Fundamental engine
# -----------------------------

def fundamental_engine(data):
    ai = data["annual_income"]
    ab = data["annual_balance"]
    ac = data["annual_cashflow"]
    qi = data["quarterly_income"]
    qb = data["quarterly_balance"]
    qc = data["quarterly_cashflow"]

    revenue = latest_value(ai, ["Total Revenue", "Operating Revenue"])
    revenue_prev = previous_value(ai, ["Total Revenue", "Operating Revenue"])
    rev_growth = growth(revenue, revenue_prev)

    ebit = latest_value(ai, ["EBIT", "Operating Income"])
    ebit_prev = previous_value(ai, ["EBIT", "Operating Income"])
    ebit_growth = growth(ebit, ebit_prev)

    net_income = latest_value(ai, ["Net Income", "Net Income Common Stockholders"])
    net_income_prev = previous_value(ai, ["Net Income", "Net Income Common Stockholders"])
    earnings_growth = growth(net_income, net_income_prev)

    equity = latest_value(ab, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    debt = latest_value(ab, ["Total Debt", "Total Debt And Capital Lease Obligation"])
    cash = latest_value(ab, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    assets = latest_value(ab, ["Total Assets"])

    roe = net_income / equity if not pd.isna(net_income) and not pd.isna(equity) and equity != 0 else np.nan
    roa = net_income / assets if not pd.isna(net_income) and not pd.isna(assets) and assets != 0 else np.nan
    de = debt / equity if not pd.isna(debt) and not pd.isna(equity) and equity != 0 else np.nan

    cfo = latest_value(ac, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = latest_value(ac, ["Capital Expenditure", "Capital Expenditures"])
    fcf = latest_value(ac, ["Free Cash Flow"])
    if pd.isna(fcf) and not pd.isna(cfo) and not pd.isna(capex):
        fcf = cfo + capex if capex < 0 else cfo - capex

    cfo_to_pat = cfo / net_income if not pd.isna(cfo) and not pd.isna(net_income) and net_income != 0 else np.nan

    # Latest quarter and previous quarter
    q_revenue = latest_value(qi, ["Total Revenue", "Operating Revenue"])
    q_revenue_prev = previous_value(qi, ["Total Revenue", "Operating Revenue"])
    q_earnings = latest_value(qi, ["Net Income", "Net Income Common Stockholders"])
    q_earnings_prev = previous_value(qi, ["Net Income", "Net Income Common Stockholders"])

    q_revenue_growth_seq = growth(q_revenue, q_revenue_prev)
    q_earnings_growth_seq = growth(q_earnings, q_earnings_prev)

    score_parts = {
        "Revenue growth": score_higher_better(rev_growth, 0.10, 0.20),
        "Earnings growth": score_higher_better(earnings_growth, 0.10, 0.20),
        "ROE": score_higher_better(roe, 0.10, 0.20),
        "Debt/equity": score_lower_better(de, 1.5, 0.5),
        "CFO/PAT": score_higher_better(cfo_to_pat, 0.8, 1.2),
        "ROA": score_higher_better(roa, 0.03, 0.10),
    }

    fundamental_score = float(np.mean(list(score_parts.values())))

    positives, concerns = [], []

    if not pd.isna(rev_growth) and rev_growth > 0.10:
        positives.append(f"Annual revenue growth is {pct(rev_growth)}.")
    elif not pd.isna(rev_growth) and rev_growth < 0:
        concerns.append(f"Annual revenue declined {pct(abs(rev_growth))}.")

    if not pd.isna(roe):
        (positives if roe >= 0.15 else concerns).append(f"ROE is {pct(roe)}.")

    if not pd.isna(de):
        (positives if de <= 0.5 else concerns).append(f"Debt/equity is {de:.2f}.")

    if not pd.isna(cfo_to_pat):
        if cfo_to_pat >= 1:
            positives.append(f"Operating cash flow is {cfo_to_pat:.2f}x reported PAT.")
        elif cfo_to_pat < 0.7:
            concerns.append(f"Cash conversion is weak at {cfo_to_pat:.2f}x PAT.")

    if not pd.isna(fcf):
        (positives if fcf > 0 else concerns).append(
            f"Latest annual free cash flow is {money(fcf)}."
        )

    return {
        "score": fundamental_score,
        "score_parts": score_parts,
        "revenue": revenue,
        "rev_growth": rev_growth,
        "ebit": ebit,
        "ebit_growth": ebit_growth,
        "net_income": net_income,
        "earnings_growth": earnings_growth,
        "equity": equity,
        "debt": debt,
        "cash": cash,
        "roe": roe,
        "roa": roa,
        "de": de,
        "cfo": cfo,
        "fcf": fcf,
        "cfo_to_pat": cfo_to_pat,
        "q_revenue": q_revenue,
        "q_revenue_growth_seq": q_revenue_growth_seq,
        "q_earnings": q_earnings,
        "q_earnings_growth_seq": q_earnings_growth_seq,
        "positives": positives,
        "concerns": concerns,
        "quarterly_income": qi,
        "quarterly_balance": qb,
        "quarterly_cashflow": qc,
    }


# -----------------------------
# Valuation engine
# -----------------------------

def valuation_engine(info, price, fundamental):
    pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    pb = safe_float(info.get("priceToBook"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
    dividend_yield = safe_float(info.get("dividendYield"))

    scores = []

    # These are deliberately conservative because sector multiples differ.
    if not pd.isna(pe):
        scores.append(80 if pe < 15 else 65 if pe < 22 else 50 if pe < 30 else 30)
    if not pd.isna(forward_pe):
        scores.append(80 if forward_pe < 15 else 65 if forward_pe < 22 else 50 if forward_pe < 30 else 30)
    if not pd.isna(pb):
        scores.append(80 if pb < 3 else 60 if pb < 5 else 40 if pb < 8 else 25)
    if not pd.isna(ev_ebitda):
        scores.append(80 if ev_ebitda < 12 else 65 if ev_ebitda < 18 else 50 if ev_ebitda < 25 else 30)

    score = float(np.mean(scores)) if scores else 50.0

    positives, concerns = [], []
    if not pd.isna(forward_pe) and not pd.isna(pe) and forward_pe < pe:
        positives.append("Forward P/E is below trailing P/E, implying expected earnings improvement.")
    if not pd.isna(pe) and pe > 30:
        concerns.append(f"Trailing P/E is elevated at {pe:.1f}x.")
    if not pd.isna(ev_ebitda) and ev_ebitda > 25:
        concerns.append(f"EV/EBITDA is elevated at {ev_ebitda:.1f}x.")

    return {
        "score": score,
        "pe": pe,
        "forward_pe": forward_pe,
        "pb": pb,
        "ps": ps,
        "ev_ebitda": ev_ebitda,
        "dividend_yield": dividend_yield,
        "positives": positives,
        "concerns": concerns,
    }


# -----------------------------
# Quarterly engine
# -----------------------------

def quarterly_engine(fundamental):
    qi = fundamental["quarterly_income"]
    if qi is None or qi.empty:
        return {"score": 50, "table": pd.DataFrame(), "commentary": ["Quarterly data unavailable."]}

    rows = {}
    for label, names in {
        "Revenue": ["Total Revenue", "Operating Revenue"],
        "EBIT": ["EBIT", "Operating Income"],
        "PAT": ["Net Income", "Net Income Common Stockholders"],
        "EBITDA": ["EBITDA"],
    }.items():
        if any(n in qi.index for n in names):
            for n in names:
                if n in qi.index:
                    rows[label] = pd.to_numeric(qi.loc[n], errors="coerce")
                    break

    if not rows:
        return {"score": 50, "table": pd.DataFrame(), "commentary": ["Quarterly income statement unavailable."]}

    qtable = pd.DataFrame(rows)
    qtable.index = pd.to_datetime(qtable.index).strftime("%Y-%m-%d")
    qtable = qtable.sort_index(ascending=False)

    comments = []
    score = 50.0

    if "Revenue" in qtable:
        vals = qtable["Revenue"].dropna()
        if len(vals) >= 2:
            g = growth(vals.iloc[0], vals.iloc[1])
            if not pd.isna(g):
                if g > 0.05:
                    score += 12
                    comments.append(f"Latest quarter revenue grew {pct(g)} sequentially.")
                elif g < -0.05:
                    score -= 12
                    comments.append(f"Latest quarter revenue declined {pct(abs(g))} sequentially.")

    if "PAT" in qtable:
        vals = qtable["PAT"].dropna()
        if len(vals) >= 2:
            g = growth(vals.iloc[0], vals.iloc[1])
            if not pd.isna(g):
                if g > 0.05:
                    score += 12
                    comments.append(f"Latest quarter PAT grew {pct(g)} sequentially.")
                elif g < -0.05:
                    score -= 12
                    comments.append(f"Latest quarter PAT declined {pct(abs(g))} sequentially.")

    return {
        "score": max(0, min(100, score)),
        "table": qtable,
        "commentary": comments,
    }


# -----------------------------
# Risk + position sizing
# -----------------------------

def risk_engine(hist, capital, risk_pct, objective):
    d = add_indicators(hist)
    r = d.iloc[-1]

    price = safe_float(r["Close"])
    atr = safe_float(r["ATR14"])
    sma20 = safe_float(r["SMA20"])
    sma50 = safe_float(r["SMA50"])

    if pd.isna(atr) or atr <= 0:
        atr = price * 0.03

    if objective == "Long Term":
        stop = min(
            price - 2.5 * atr,
            sma50 * 0.93 if not pd.isna(sma50) else price - 2.5 * atr,
        )
        risk_multiple = 3.0
    elif objective == "Swing":
        stop = min(
            price - 1.8 * atr,
            sma20 * 0.96 if not pd.isna(sma20) else price - 1.8 * atr,
        )
        risk_multiple = 2.5
    else:
        stop = price - 1.2 * atr
        risk_multiple = 1.5

    if stop <= 0 or stop >= price:
        stop = price * 0.95

    risk_per_share = price - stop
    max_loss = capital * risk_pct / 100
    qty = int(max_loss / risk_per_share) if risk_per_share > 0 else 0
    deployed = qty * price
    target = price + risk_multiple * risk_per_share

    return {
        "price": price,
        "atr": atr,
        "stop": stop,
        "risk_per_share": risk_per_share,
        "max_loss": max_loss,
        "qty": qty,
        "deployed": deployed,
        "target": target,
        "rr": risk_multiple,
    }


# -----------------------------
# Decision engine
# -----------------------------

def decision_engine(fundamental, quarterly, valuation, technical, objective):
    weights = {
        "Long Term": {"fundamental": 0.35, "quarterly": 0.20, "valuation": 0.20, "technical": 0.15, "quality": 0.10},
        "Swing": {"fundamental": 0.15, "quarterly": 0.15, "valuation": 0.10, "technical": 0.50, "quality": 0.10},
        "Intraday": {"fundamental": 0.05, "quarterly": 0.05, "valuation": 0.05, "technical": 0.75, "quality": 0.10},
    }[objective]

    quality = np.mean([
        fundamental["score_parts"].get("ROE", 50),
        fundamental["score_parts"].get("Debt/equity", 50),
        fundamental["score_parts"].get("CFO/PAT", 50),
    ])

    score = (
        fundamental["score"] * weights["fundamental"]
        + quarterly["score"] * weights["quarterly"]
        + valuation["score"] * weights["valuation"]
        + technical * weights["technical"]
        + quality * weights["quality"]
    )

    if score >= 75:
        action = "ACCUMULATE / BUY ON CONFIRMATION"
    elif score >= 62:
        action = "SELECTIVE ENTRY / WATCH"
    elif score >= 48:
        action = "HOLD / WAIT FOR BETTER SETUP"
    else:
        action = "AVOID / WAIT"

    return float(score), action


# -----------------------------
# UI
# -----------------------------

st.title("📊 Indian Stock Intelligence — V2.1")
st.caption("Fundamental + Quarterly + Valuation + Technical + Risk decision-support engine")

st.info(
    "V2.1 uses publicly available Yahoo Finance data through yfinance. "
    "Data can be delayed, incomplete or differently mapped from company filings. "
    "Management guidance and earnings-call commentary are NOT invented when unavailable. "
    "Verify material decisions against NSE/BSE/company filings."
)

with st.sidebar:
    st.header("Analysis Inputs")

    symbol = st.text_input("NSE stock symbol", value="RELIANCE").strip().upper()
    symbol = symbol.replace(".NS", "") + ".NS"

    capital = st.number_input(
        "Portfolio capital (₹)",
        min_value=10000,
        value=500000,
        step=10000,
    )

    risk_pct = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    objective = st.selectbox(
        "Primary objective",
        ["Long Term", "Swing", "Intraday"],
        index=0,
    )

    run = st.button("🚀 RUN V2.1 ANALYSIS", use_container_width=True, type="primary")

if not run:
    st.markdown("## What V2.1 adds")
    st.markdown(
        """
        - **Fundamental quality score:** growth, ROE, ROA, leverage and cash conversion
        - **Quarterly trend engine:** sequential revenue/PAT direction
        - **Valuation engine:** P/E, forward P/E, P/B, P/S and EV/EBITDA where available
        - **Technical engine:** 20/50/100/200 DMA, RSI, MACD, ADX, ATR and volume
        - **Entry framework:** support/stop/target context
        - **Risk-based position sizing:** capital × risk % ÷ stop-loss distance
        - **Separate Long Term / Swing / Intraday weighting**
        - **Management evidence discipline:** unavailable guidance is explicitly marked rather than invented
        """
    )
    st.warning("Enter an NSE symbol and click RUN V2.1 ANALYSIS.")
    st.stop()

try:
    data = load_stock(symbol)
except Exception as e:
    st.error(f"Could not analyse {symbol.replace('.NS','')}: {e}")
    st.stop()

hist = data["hist"]
hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
d = add_indicators(hist)

fundamental = fundamental_engine(data)
quarterly = quarterly_engine(fundamental)
valuation = valuation_engine(data["info"], safe_float(d["Close"].iloc[-1]), fundamental)
technical_score_value, technical_notes = technical_score(d)
risk = risk_engine(hist, capital, risk_pct, objective)
overall, action = decision_engine(
    fundamental,
    quarterly,
    valuation,
    technical_score_value,
    objective,
)

# -----------------------------
# Header metrics
# -----------------------------

st.subheader(f"{data['info'].get('longName', symbol.replace('.NS',''))} ({symbol.replace('.NS','')})")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("V2.1 Score", f"{overall:.0f}/100")
m2.metric("Fundamental", f"{fundamental['score']:.0f}")
m3.metric("Quarterly", f"{quarterly['score']:.0f}")
m4.metric("Valuation", f"{valuation['score']:.0f}")
m5.metric("Technical", f"{technical_score_value:.0f}")

if overall >= 75:
    st.success(f"🟢 {action}")
elif overall >= 62:
    st.warning(f"🟡 {action}")
else:
    st.error(f"🔴 {action}")

st.caption(f"Primary objective: {objective}")

# -----------------------------
# Price / market snapshot
# -----------------------------

st.markdown("## Market snapshot")
s1, s2, s3, s4, s5 = st.columns(5)

price = safe_float(d["Close"].iloc[-1])
high52 = safe_float(d["52W_High"].iloc[-1])
low52 = safe_float(d["52W_Low"].iloc[-1])
market_cap = safe_float(data["info"].get("marketCap"))
beta = safe_float(data["info"].get("beta"))

s1.metric("Price", money(price))
s2.metric("52W High", money(high52))
s3.metric("52W Low", money(low52))
s4.metric("Market Cap", money(market_cap))
s5.metric("Beta", "N/A" if pd.isna(beta) else f"{beta:.2f}")

# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs([
    "🎯 Executive Decision",
    "🏢 Fundamentals",
    "📅 Quarterly",
    "💰 Valuation",
    "📈 Technicals",
    "🛡️ Risk & Position",
    "🧠 Management / Evidence",
    "📰 Recent News",
])

with tabs[0]:
    st.markdown("### Decision framework")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Key positives")
        positives = fundamental["positives"] + valuation["positives"]
        positives += [x for x in technical_notes if not any(k in x.lower() for k in ["below", "weak", "overbought"])]
        if positives:
            for x in positives[:8]:
                st.success("✓ " + x)
        else:
            st.info("No strong positive signal identified from available data.")

    with c2:
        st.markdown("#### Key concerns")
        concerns = fundamental["concerns"] + valuation["concerns"]
        concerns += [x for x in technical_notes if any(k in x.lower() for k in ["below", "weak", "overbought"])]
        if concerns:
            for x in concerns[:8]:
                st.error("⚠ " + x)
        else:
            st.success("No major automated red flag identified.")

    st.markdown("### What would improve the setup?")
    if technical_score_value < 60:
        st.write("• Wait for technical trend confirmation: price above key moving averages and improving momentum.")
    if fundamental["score"] < 60:
        st.write("• Wait for improvement in earnings quality, cash conversion, leverage or growth.")
    if valuation["score"] < 60:
        st.write("• Prefer an improved valuation/price entry rather than chasing the stock.")
    if technical_score_value >= 60 and fundamental["score"] >= 60 and valuation["score"] >= 60:
        st.write("• The setup is reasonably aligned; use the risk framework rather than deploying the full portfolio at once.")

with tabs[1]:
    st.markdown("### Fundamental quality")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Revenue growth", pct(fundamental["rev_growth"]))
    f2.metric("EBIT growth", pct(fundamental["ebit_growth"]))
    f3.metric("PAT growth", pct(fundamental["earnings_growth"]))
    f4.metric("ROE", pct(fundamental["roe"]))

    f5, f6, f7, f8 = st.columns(4)
    f5.metric("ROA", pct(fundamental["roa"]))
    f6.metric("Debt / Equity", "N/A" if pd.isna(fundamental["de"]) else f"{fundamental['de']:.2f}x")
    f7.metric("CFO / PAT", "N/A" if pd.isna(fundamental["cfo_to_pat"]) else f"{fundamental['cfo_to_pat']:.2f}x")
    f8.metric("Free cash flow", money(fundamental["fcf"]))

    st.markdown("### Fundamental score components")
    score_df = pd.DataFrame(
        {"Score": fundamental["score_parts"]}
    )
    st.bar_chart(score_df)

    st.markdown("### Annual income statement")
    ai = data["annual_income"]
    if ai is not None and not ai.empty:
        display_rows = [r for r in ["Total Revenue", "EBIT", "EBITDA", "Net Income", "Diluted EPS"] if r in ai.index]
        if display_rows:
            st.dataframe(ai.loc[display_rows], use_container_width=True)
    else:
        st.info("Annual income statement unavailable.")

with tabs[2]:
    st.markdown("### Quarterly trend")
    if not quarterly["table"].empty:
        st.dataframe(quarterly["table"].head(8), use_container_width=True)
        for c in quarterly["commentary"]:
            st.info("• " + c)
    else:
        st.warning("Quarterly data is unavailable from the current data source.")

with tabs[3]:
    st.markdown("### Valuation")
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("P/E", "N/A" if pd.isna(valuation["pe"]) else f"{valuation['pe']:.1f}x")
    v2.metric("Forward P/E", "N/A" if pd.isna(valuation["forward_pe"]) else f"{valuation['forward_pe']:.1f}x")
    v3.metric("P/B", "N/A" if pd.isna(valuation["pb"]) else f"{valuation['pb']:.1f}x")
    v4.metric("P/S", "N/A" if pd.isna(valuation["ps"]) else f"{valuation['ps']:.1f}x")
    v5.metric("EV/EBITDA", "N/A" if pd.isna(valuation["ev_ebitda"]) else f"{valuation['ev_ebitda']:.1f}x")

    if valuation["dividend_yield"] and not pd.isna(valuation["dividend_yield"]):
        st.write(f"Dividend yield: **{pct(valuation['dividend_yield'])}**")

    st.warning(
        "Valuation is deliberately not sector-normalized in V2.1. "
        "Use peer multiples before treating this score as a buy/sell signal."
    )

with tabs[4]:
    st.markdown("### Technical dashboard")

    r = d.iloc[-1]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("RSI(14)", "N/A" if pd.isna(r["RSI14"]) else f"{r['RSI14']:.1f}")
    t2.metric("ADX(14)", "N/A" if pd.isna(r["ADX14"]) else f"{r['ADX14']:.1f}")
    t3.metric("ATR", money(r["ATR14"]))
    t4.metric("Volume / 20D avg", "N/A" if pd.isna(r["VolRatio"]) else f"{r['VolRatio']:.2f}x")

    chart_cols = ["Close", "SMA20", "SMA50", "SMA200"]
    chart_df = d[chart_cols].tail(300).copy()
    st.line_chart(chart_df)

    tech_table = pd.DataFrame({
        "Indicator": [
            "Close", "20 DMA", "50 DMA", "100 DMA", "200 DMA",
            "RSI14", "MACD", "MACD Signal", "ADX14", "ATR14",
            "52W High", "52W Low", "20D Return", "60D Return"
        ],
        "Value": [
            r["Close"], r["SMA20"], r["SMA50"], r["SMA100"], r["SMA200"],
            r["RSI14"], r["MACD"], r["MACD_signal"], r["ADX14"], r["ATR14"],
            r["52W_High"], r["52W_Low"], r["Return20"], r["Return60"]
        ]
    })
    st.dataframe(tech_table, use_container_width=True)

    st.markdown("### Technical interpretation")
    for note in technical_notes:
        st.write("• " + note)

with tabs[5]:
    st.markdown("### Risk-based position sizing")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Entry/reference price", money(risk["price"]))
    r2.metric("Suggested stop", money(risk["stop"]))
    r3.metric("Max loss", money(risk["max_loss"]))
    r4.metric("Risk / share", money(risk["risk_per_share"]))

    r5, r6, r7 = st.columns(3)
    r5.metric("Suggested quantity", f"{risk['qty']:,}")
    r6.metric("Capital deployed", money(risk["deployed"]))
    r7.metric("Illustrative target", money(risk["target"]))

    st.write(
        f"Risk/reward framework: approximately **1:{risk['rr']:.1f}** "
        f"for the selected **{objective}** objective."
    )

    st.warning(
        "Position sizing is a risk-control calculation, not a recommendation to buy. "
        "Actual stop placement should be checked against chart structure/support."
    )

with tabs[6]:
    st.markdown("### Management / Evidence discipline")

    st.info(
        "V2.1 does not manufacture management guidance. "
        "Yahoo Finance/yfinance does not reliably provide complete earnings-call transcripts, "
        "investor-presentation guidance or company-specific FY27/FY28 targets."
    )

    st.markdown("#### What is currently verified by structured data")
    st.write("• Financial statements and market data returned by yfinance.")
    st.write("• Current/last available price and technical indicators calculated from price history.")
    st.write("• Valuation fields only where Yahoo Finance supplies them.")

    st.markdown("#### What requires filing-level verification")
    st.write("• FY27/FY28 revenue or EBITDA guidance")
    st.write("• Order-book figures")
    st.write("• Capacity expansion / capex commitments")
    st.write("• Management commentary")
    st.write("• New customer wins and project pipelines")
    st.write("• Segment-level future targets")

    st.warning(
        "For serious investment decisions, use company annual reports, quarterly investor presentations, "
        "earnings-call transcripts and NSE/BSE filings to supplement this screen."
    )

with tabs[7]:
    st.markdown("### Recent news / evidence")
    news = data.get("news", [])

    if not news:
        st.info("No recent news items were returned by the data source.")
    else:
        shown = 0
        for item in news[:10]:
            content = item.get("content", item)
            if not isinstance(content, dict):
                continue

            title = content.get("title") or item.get("title") or "Untitled"
            publisher = content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else ""
            link = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("link", "")

            st.markdown(f"**{title}**")
            if publisher:
                st.caption(publisher)
            if link:
                st.markdown(f"[Open article]({link})")
            shown += 1

        if shown == 0:
            st.info("News was returned but could not be parsed into displayable items.")

# -----------------------------
# Footer
# -----------------------------

st.divider()
st.caption(
    f"Indian Stock Intelligence V2.1 • Data checked: {datetime.now().strftime('%d-%b-%Y %H:%M')} • "
    "Decision-support only; not investment advice."
)
