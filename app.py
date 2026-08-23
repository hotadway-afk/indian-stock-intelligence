
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Indian Stock Intelligence V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------
def safe_num(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def fmt_num(x, decimals=1):
    x = safe_num(x)
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"

def fmt_pct(x, decimals=1):
    x = safe_num(x)
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.{decimals}f}%"

def score_range(x, low, high, inverse=False):
    x = safe_num(x)
    if pd.isna(x):
        return 50.0
    s = float(np.clip((x-low)/(high-low)*100, 0, 100))
    return 100-s if inverse else s

def normalize_statement(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    # yfinance often uses timestamps as columns; convert them to readable dates
    try:
        out.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") for c in out.columns]
    except Exception:
        out.columns = [str(c) for c in out.columns]
    return out

def latest_row(df, names):
    if df is None or df.empty:
        return np.nan
    idx = {str(i).lower(): i for i in df.index}
    for n in names:
        key = n.lower()
        if key in idx:
            row = df.loc[idx[key]]
            try:
                return safe_num(row.iloc[0])
            except Exception:
                pass
    return np.nan

def growth_from_statement(df, names):
    if df is None or df.empty:
        return np.nan
    idx = {str(i).lower(): i for i in df.index}
    for n in names:
        key = n.lower()
        if key in idx:
            row = pd.to_numeric(df.loc[idx[key]], errors="coerce")
            vals = row.dropna()
            if len(vals) >= 2:
                first = vals.iloc[-1]
                last = vals.iloc[0]
                if first != 0:
                    return (last/first) ** (1/(len(vals)-1)) - 1
    return np.nan

def find_row(df, candidates):
    if df is None or df.empty:
        return None
    lower = {str(i).lower(): i for i in df.index}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for i in df.index:
        s = str(i).lower()
        for c in candidates:
            if c.lower() in s:
                return i
    return None

# -----------------------------
# Data layer
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_stock(symbol):
    ticker = yf.Ticker(symbol + ".NS")

    hist = ticker.history(period="5y", auto_adjust=False)
    info = dict(ticker.info or {})

    annual_income = normalize_statement(ticker.get_income_stmt(freq="yearly"))
    quarterly_income = normalize_statement(ticker.get_income_stmt(freq="quarterly"))
    annual_balance = normalize_statement(ticker.get_balance_sheet(freq="yearly"))
    quarterly_balance = normalize_statement(ticker.get_balance_sheet(freq="quarterly"))
    annual_cashflow = normalize_statement(ticker.get_cash_flow(freq="yearly"))
    quarterly_cashflow = normalize_statement(ticker.get_cash_flow(freq="quarterly"))

    # News is converted to simple dictionaries, not Ticker objects.
    try:
        raw_news = ticker.news or []
        news = []
        for item in raw_news[:20]:
            content = item.get("content", item)
            if isinstance(content, dict):
                title = content.get("title") or item.get("title")
                publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
                url = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link")
                pub = content.get("pubDate") or item.get("providerPublishTime")
            else:
                title = item.get("title")
                publisher = item.get("publisher")
                url = item.get("link")
                pub = item.get("providerPublishTime")
            news.append({"title": title, "publisher": publisher, "url": url, "published": pub})
    except Exception:
        news = []

    return {
        "hist": hist,
        "info": info,
        "annual_income": annual_income,
        "quarterly_income": quarterly_income,
        "annual_balance": annual_balance,
        "quarterly_balance": quarterly_balance,
        "annual_cashflow": annual_cashflow,
        "quarterly_cashflow": quarterly_cashflow,
        "news": news,
    }

# -----------------------------
# Technical engine
# -----------------------------
def technical_engine(hist):
    h = hist.copy()
    close = pd.to_numeric(h["Close"], errors="coerce")
    high = pd.to_numeric(h["High"], errors="coerce")
    low = pd.to_numeric(h["Low"], errors="coerce")
    volume = pd.to_numeric(h["Volume"], errors="coerce")

    for n in [20, 50, 100, 200]:
        h[f"SMA{n}"] = close.rolling(n).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    h["MACD"] = ema12 - ema26
    h["MACDSignal"] = h["MACD"].ewm(span=9, adjust=False).mean()
    h["MACDHist"] = h["MACD"] - h["MACDSignal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    h["RSI14"] = 100 - 100/(1+rs)

    tr = pd.concat([
        high-low,
        (high-close.shift()).abs(),
        (low-close.shift()).abs()
    ], axis=1).max(axis=1)
    h["ATR14"] = tr.rolling(14).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    h["BBMid"] = mid
    h["BBUpper"] = mid + 2*std
    h["BBLower"] = mid - 2*std

    h["Vol20"] = volume.rolling(20).mean()
    h["VolumeRatio"] = volume / h["Vol20"]

    # ADX
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).rolling(14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).rolling(14).mean() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    h["ADX14"] = dx.rolling(14).mean()

    latest = h.iloc[-1]
    price = safe_num(latest["Close"])

    score = 50.0
    components = {}

    components["Price > 20DMA"] = 15 if price > safe_num(latest["SMA20"]) else 0
    components["Price > 50DMA"] = 15 if price > safe_num(latest["SMA50"]) else 0
    components["Price > 200DMA"] = 20 if price > safe_num(latest["SMA200"]) else 0
    components["50DMA > 200DMA"] = 15 if safe_num(latest["SMA50"]) > safe_num(latest["SMA200"]) else 0
    components["MACD bullish"] = 10 if safe_num(latest["MACD"]) > safe_num(latest["MACDSignal"]) else 0
    components["RSI healthy"] = 10 if 50 <= safe_num(latest["RSI14"]) <= 70 else (5 if 40 <= safe_num(latest["RSI14"]) < 50 else 0)
    components["ADX > 20"] = 10 if safe_num(latest["ADX14"]) > 20 else 0
    components["Volume expansion"] = 5 if safe_num(latest["VolumeRatio"]) > 1.2 else 0

    score = sum(components.values())
    return h, score, components

# -----------------------------
# Fundamental engine
# -----------------------------
def fundamental_engine(info, annual_income, annual_balance, annual_cashflow):
    revenue = latest_row(annual_income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    net_income = latest_row(annual_income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    ebitda = latest_row(annual_income, ["EBITDA", "Normalized EBITDA"])
    operating_cf = latest_row(annual_cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
    capex = latest_row(annual_cashflow, ["Capital Expenditure", "Capital Expenditures"])
    debt = latest_row(annual_balance, ["Total Debt", "Total Debt And Capital Lease Obligation"])
    equity = latest_row(annual_balance, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    cash = latest_row(annual_balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])

    roe = safe_num(info.get("returnOnEquity"))
    roa = safe_num(info.get("returnOnAssets"))
    margin = safe_num(info.get("profitMargins"))
    rev_growth = safe_num(info.get("revenueGrowth"))
    earnings_growth = safe_num(info.get("earningsGrowth"))
    de = safe_num(info.get("debtToEquity"))

    # Approximate FCF from annual cashflow when available
    fcf = np.nan
    if not pd.isna(operating_cf) and not pd.isna(capex):
        fcf = operating_cf + capex if capex < 0 else operating_cf - capex

    components = {
        "ROE": score_range(roe, 0.08, 0.25),
        "ROA": score_range(roa, 0.03, 0.15),
        "Revenue growth": score_range(rev_growth, 0.00, 0.25),
        "Earnings growth": score_range(earnings_growth, 0.00, 0.30),
        "Profit margin": score_range(margin, 0.05, 0.25),
        "Debt discipline": score_range(de, 20, 150, inverse=True),
    }

    score = float(np.mean(list(components.values())))

    return {
        "score": score,
        "components": components,
        "revenue": revenue,
        "net_income": net_income,
        "ebitda": ebitda,
        "operating_cf": operating_cf,
        "capex": capex,
        "fcf": fcf,
        "debt": debt,
        "equity": equity,
        "cash": cash,
        "roe": roe,
        "roa": roa,
        "margin": margin,
        "rev_growth": rev_growth,
        "earnings_growth": earnings_growth,
        "de": de,
    }

# -----------------------------
# Valuation engine
# -----------------------------
def valuation_engine(info, fundamental):
    pe = safe_num(info.get("trailingPE"))
    fpe = safe_num(info.get("forwardPE"))
    peg = safe_num(info.get("pegRatio"))
    pb = safe_num(info.get("priceToBook"))
    ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
    ps = safe_num(info.get("priceToSalesTrailing12Months"))

    parts = {
        "P/E": 50 if pd.isna(pe) else score_range(pe, 10, 60, inverse=True),
        "Forward P/E": 50 if pd.isna(fpe) else score_range(fpe, 8, 50, inverse=True),
        "PEG": 50 if pd.isna(peg) else score_range(peg, 0.5, 3.0, inverse=True),
        "P/B": 50 if pd.isna(pb) else score_range(pb, 1, 10, inverse=True),
        "EV/EBITDA": 50 if pd.isna(ev_ebitda) else score_range(ev_ebitda, 5, 35, inverse=True),
    }

    score = float(np.mean(list(parts.values())))
    return {
        "score": score,
        "parts": parts,
        "pe": pe,
        "forward_pe": fpe,
        "peg": peg,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "ps": ps,
    }

# -----------------------------
# Quality / red-flag engine
# -----------------------------
def quality_engine(info, fundamental, annual_income, annual_cashflow, annual_balance):
    flags = []
    positives = []

    if not pd.isna(fundamental["roe"]) and fundamental["roe"] >= 0.15:
        positives.append("ROE is above 15% based on available data.")
    if not pd.isna(fundamental["rev_growth"]) and fundamental["rev_growth"] > 0.10:
        positives.append("Revenue growth is above 10% in the available company snapshot.")
    if not pd.isna(fundamental["fcf"]) and fundamental["fcf"] > 0:
        positives.append("Latest annual cash flow indicates positive free cash flow.")

    if not pd.isna(fundamental["de"]) and fundamental["de"] > 150:
        flags.append("Debt/equity is elevated.")
    if not pd.isna(fundamental["rev_growth"]) and fundamental["rev_growth"] < 0:
        flags.append("Revenue growth is negative.")
    if not pd.isna(fundamental["earnings_growth"]) and fundamental["earnings_growth"] < 0:
        flags.append("Earnings growth is negative.")
    if not pd.isna(fundamental["fcf"]) and not pd.isna(fundamental["net_income"]):
        if fundamental["fcf"] < 0 and fundamental["net_income"] > 0:
            flags.append("Positive accounting profit with negative latest annual free cash flow; investigate earnings quality/capex.")

    promoter = safe_num(info.get("heldPercentInsiders"))
    if not pd.isna(promoter) and promoter < 0.20:
        flags.append("Insider/promoter ownership is below 20% in the available snapshot.")

    score = max(0, 100 - 12*len(flags) + 6*len(positives))
    return score, positives, flags

# -----------------------------
# Main UI
# -----------------------------
st.title("📊 Indian Stock Intelligence — V2")
st.caption("Fundamental + Management/Evidence + Valuation + Technical + Risk decision-support engine")

with st.sidebar:
    st.header("Analysis Inputs")
    stock = st.text_input("NSE stock symbol", "RELIANCE").strip().upper()
    capital = st.number_input("Portfolio capital (₹)", min_value=10000, value=500000, step=10000)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    objective = st.selectbox("Primary objective", ["Long Term", "Swing Trading", "Intraday"])
    run = st.button("🚀 RUN V2 ANALYSIS", type="primary", use_container_width=True)

if not run:
    st.info("Enter an NSE symbol and click RUN V2 ANALYSIS.")
    st.markdown("""
### What V2 adds
- 5-year price history and multi-indicator technical engine
- Annual + quarterly income statement, balance sheet and cash-flow views
- Fundamental quality scoring
- Valuation scoring
- Red-flag and positive-catalyst checks from available structured data
- Recent news/evidence panel
- Risk-based position sizing
- Separate long-term / swing / intraday context

**Important:** V2 is a decision-support system, not a guaranteed-return or automated trading system.
Management guidance and filing-level verification are clearly marked when the free data layer cannot establish them.
""")
    st.stop()

try:
    with st.spinner(f"Loading {stock} data..."):
        data = load_stock(stock)

    hist = data["hist"]
    info = data["info"]

    if hist.empty:
        st.error(f"No NSE market data returned for {stock}. Check the ticker symbol.")
        st.stop()

    tech, technical_score, technical_components = technical_engine(hist)
    fundamental = fundamental_engine(
        info, data["annual_income"], data["annual_balance"], data["annual_cashflow"]
    )
    valuation = valuation_engine(info, fundamental)
    quality_score, positives, flags = quality_engine(
        info, fundamental, data["annual_income"], data["annual_cashflow"], data["annual_balance"]
    )

    # Management/evidence score is intentionally conservative.
    # Without filing/earnings-call text, do not fabricate a management-quality score.
    management_score = 50.0
    evidence_note = "Structured free-data layer loaded. Management guidance/commentary has not been independently verified from company filings in V2."

    # Overall weighted score
    overall = (
        0.35 * fundamental["score"]
        + 0.20 * management_score
        + 0.15 * valuation["score"]
        + 0.20 * technical_score
        + 0.10 * quality_score
    )

    if objective == "Long Term":
        objective_score = 0.45*fundamental["score"] + 0.25*management_score + 0.20*valuation["score"] + 0.10*quality_score
    elif objective == "Swing Trading":
        objective_score = 0.55*technical_score + 0.20*valuation["score"] + 0.15*quality_score + 0.10*fundamental["score"]
    else:
        objective_score = 0.70*technical_score + 0.20*quality_score + 0.10*valuation["score"]

    if objective_score >= 75:
        verdict = "🟢 STRONG SETUP"
    elif objective_score >= 60:
        verdict = "🟡 SELECTIVE / WATCH"
    elif objective_score >= 45:
        verdict = "🟠 WAIT / NEUTRAL"
    else:
        verdict = "🔴 AVOID / HIGH RISK"

    company = info.get("longName") or stock
    price = safe_num(tech.iloc[-1]["Close"])
    high52 = safe_num(info.get("fiftyTwoWeekHigh"))
    low52 = safe_num(info.get("fiftyTwoWeekLow"))

    st.subheader(f"{company} ({stock})")

    cols = st.columns(6)
    cols[0].metric("V2 Score", f"{overall:.0f}/100")
    cols[1].metric("Fundamental", f"{fundamental['score']:.0f}")
    cols[2].metric("Management/Evidence", f"{management_score:.0f}")
    cols[3].metric("Valuation", f"{valuation['score']:.0f}")
    cols[4].metric("Technical", f"{technical_score:.0f}")
    cols[5].metric("Quality/Risk", f"{quality_score:.0f}")

    if "STRONG" in verdict:
        st.success(verdict)
    elif "WATCH" in verdict:
        st.warning(verdict)
    elif "WAIT" in verdict:
        st.info(verdict)
    else:
        st.error(verdict)

    st.caption(f"Primary objective: {objective} | Objective-specific score: {objective_score:.0f}/100")

    tabs = st.tabs([
        "🎯 Executive Decision",
        "🏢 Fundamentals",
        "🧠 Management & Evidence",
        "💰 Valuation",
        "📈 Technicals",
        "🚨 Risks & Catalysts",
        "📰 Recent News",
        "📋 Raw Financials"
    ])

    with tabs[0]:
        a,b,c,d = st.columns(4)
        a.metric("Price", f"₹{price:,.2f}")
        b.metric("52W High", f"₹{high52:,.2f}" if not pd.isna(high52) else "N/A")
        c.metric("52W Low", f"₹{low52:,.2f}" if not pd.isna(low52) else "N/A")
        d.metric("Market Cap", f"₹{info.get('marketCap')/1e7:,.0f} Cr" if info.get("marketCap") else "N/A")

        st.markdown("### Decision framework")
        st.write(
            "V2 deliberately separates structured-data evidence from management commentary. "
            "It will not invent guidance, order-book figures or management promises when those are not available."
        )

        st.markdown("### Key positives")
        if positives:
            for p in positives:
                st.success("✓ " + p)
        else:
            st.write("No strong positive flags identified by the current structured-data rules.")

        st.markdown("### Key concerns")
        if flags:
            for f in flags:
                st.error("⚠ " + f)
        else:
            st.success("No major automated red flags triggered.")

    with tabs[1]:
        st.markdown("### Fundamental scorecard")
        st.dataframe(pd.DataFrame({
            "Metric": ["ROE", "ROA", "Revenue growth", "Earnings growth", "Profit margin", "Debt/Equity", "Latest annual FCF"],
            "Value": [
                fmt_pct(fundamental["roe"]),
                fmt_pct(fundamental["roa"]),
                fmt_pct(fundamental["rev_growth"]),
                fmt_pct(fundamental["earnings_growth"]),
                fmt_pct(fundamental["margin"]),
                fmt_num(fundamental["de"]),
                f"₹{fundamental['fcf']/1e7:,.1f} Cr" if not pd.isna(fundamental["fcf"]) else "N/A"
            ]
        }), use_container_width=True, hide_index=True)

        st.markdown("### Fundamental components")
        st.bar_chart(pd.Series(fundamental["components"], name="Score"))

        st.markdown("### Annual income statement")
        st.dataframe(data["annual_income"], use_container_width=True)

        st.markdown("### Quarterly income statement")
        st.dataframe(data["quarterly_income"], use_container_width=True)

    with tabs[2]:
        st.markdown("### Management / evidence status")
        st.info(evidence_note)
        st.write(
            "For a production research model, this module should ingest company annual reports, "
            "investor presentations, earnings-call transcripts and exchange filings, then compare "
            "guidance with subsequent actual results."
        )
        st.markdown("### Structured company profile")
        profile = {
            "Company": company,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Employees": info.get("fullTimeEmployees"),
            "Country": info.get("country"),
            "Promoter/insider ownership snapshot": info.get("heldPercentInsiders"),
            "Institutional ownership snapshot": info.get("heldPercentInstitutions"),
        }
        st.dataframe(pd.DataFrame(profile.items(), columns=["Field","Value"]), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("### Relative valuation")
        st.dataframe(pd.DataFrame({
            "Metric": ["Trailing P/E", "Forward P/E", "PEG", "P/B", "EV/EBITDA", "P/S"],
            "Value": [
                fmt_num(valuation["pe"]), fmt_num(valuation["forward_pe"]),
                fmt_num(valuation["peg"]), fmt_num(valuation["pb"]),
                fmt_num(valuation["ev_ebitda"]), fmt_num(valuation["ps"])
            ]
        }), use_container_width=True, hide_index=True)
        st.bar_chart(pd.Series(valuation["parts"], name="Valuation score"))
        st.caption(
            "V2 does not present a fabricated DCF. A reliable DCF needs normalized earnings/cash-flow assumptions and explicit scenario inputs; "
            "that is planned for the next valuation module."
        )

    with tabs[4]:
        latest = tech.iloc[-1]
        st.markdown("### Trend & momentum")
        st.line_chart(tech[["Close","SMA20","SMA50","SMA100","SMA200"]].tail(300))

        a,b,c,d,e = st.columns(5)
        a.metric("RSI(14)", fmt_num(latest["RSI14"]))
        b.metric("ADX(14)", fmt_num(latest["ADX14"]))
        c.metric("ATR(14)", f"₹{fmt_num(latest['ATR14'])}")
        d.metric("MACD", fmt_num(latest["MACD"], 2))
        e.metric("Volume ratio", fmt_num(latest["VolumeRatio"], 2))

        st.markdown("### Technical components")
        st.bar_chart(pd.Series(technical_components, name="Points"))

        atr = safe_num(latest["ATR14"])
        if not pd.isna(atr):
            stop = price - 1.5*atr
            risk_per_share = max(price-stop, 0.01)
            risk_amount = capital*risk_pct/100
            qty = int(risk_amount/risk_per_share)
            target1 = price + 2*risk_per_share
            target2 = price + 3*risk_per_share
            st.markdown("### Mechanical risk framework")
            a,b,c,d = st.columns(4)
            a.metric("Illustrative stop", f"₹{stop:,.2f}")
            b.metric("Risk budget", f"₹{risk_amount:,.0f}")
            c.metric("Risk-based qty", f"{qty:,}")
            d.metric("R:R targets", f"₹{target1:,.2f} / ₹{target2:,.2f}")
            st.caption("Illustrative only; not a trade recommendation. Liquidity, gaps and slippage are not included.")

    with tabs[5]:
        st.markdown("### Automated red flags")
        if flags:
            for x in flags:
                st.error("⚠ " + x)
        else:
            st.success("No major automated red flags triggered.")

        st.markdown("### Positive catalysts / quality signals")
        if positives:
            for x in positives:
                st.success("✓ " + x)
        else:
            st.write("None triggered.")

        st.markdown("### Data limitations")
        st.write(
            "V2 cannot infer promoter pledging, auditor issues, related-party transactions, order-book quality, "
            "capex execution, regulatory events or management credibility from the structured free-data layer alone. "
            "Those require filing-level research and are intentionally not fabricated."
        )

    with tabs[6]:
        if data["news"]:
            for n in data["news"]:
                title = n.get("title") or "Untitled"
                publisher = n.get("publisher") or ""
                url = n.get("url")
                if url:
                    st.markdown(f"**{title}** — {publisher}")
                    st.markdown(f"[Open source]({url})")
                else:
                    st.markdown(f"**{title}** — {publisher}")
        else:
            st.info("No recent news was returned by the available Yahoo Finance feed.")

    with tabs[7]:
        st.markdown("### Annual balance sheet")
        st.dataframe(data["annual_balance"], use_container_width=True)
        st.markdown("### Quarterly balance sheet")
        st.dataframe(data["quarterly_balance"], use_container_width=True)
        st.markdown("### Annual cash flow")
        st.dataframe(data["annual_cashflow"], use_container_width=True)
        st.markdown("### Quarterly cash flow")
        st.dataframe(data["quarterly_cashflow"], use_container_width=True)

    st.divider()
    st.caption(
        "V2 is an analytical prototype, not investment advice. Data can be delayed, incomplete or incorrectly mapped by third-party feeds. "
        "Verify material decisions against NSE/BSE/company filings."
    )

except Exception as exc:
    st.error(f"Could not analyse {stock}. Error: {type(exc).__name__}: {exc}")
    st.exception(exc)
