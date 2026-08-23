import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Stock Intelligence V1", page_icon="📊", layout="wide")

st.title("📊 Indian Stock Intelligence — V1")
st.caption("Fundamental + Technical + Valuation + Risk decision-support engine")

st.warning(
    "V1 is a research/decision-support prototype. Free data can be incomplete or delayed. "
    "Always verify important figures against NSE/BSE/company filings before investing."
)

with st.sidebar:
    st.header("Inputs")
    stock = st.text_input("NSE stock symbol", "RELIANCE").strip().upper()
    capital = st.number_input("Portfolio capital (₹)", min_value=10000, value=500000, step=10000)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    horizon = st.selectbox("Primary objective", ["Long Term", "Swing Trading", "Intraday"])
    analyse = st.button("🚀 ANALYSE STOCK", type="primary", use_container_width=True)

def score_range(x, lo, hi):
    if pd.isna(x): return 50
    return float(np.clip((x-lo)/(hi-lo)*100, 0, 100))

@st.cache_data(ttl=900)
def get_data(symbol):
    ticker = yf.Ticker(symbol + ".NS")
    hist = ticker.history(period="2y", auto_adjust=False)
    info = ticker.info
    return ticker, hist, info

def technicals(hist):
    h = hist.copy()
    close = h["Close"]
    h["SMA20"] = close.rolling(20).mean()
    h["SMA50"] = close.rolling(50).mean()
    h["SMA200"] = close.rolling(200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    h["RSI14"] = 100 - 100/(1+rs)
    h["Vol20"] = h["Volume"].rolling(20).mean()
    h["ATR14"] = pd.concat([
        h["High"]-h["Low"],
        (h["High"]-close.shift()).abs(),
        (h["Low"]-close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    return h

def fundamental_score(info):
    roe = info.get("returnOnEquity")
    de = info.get("debtToEquity")
    margin = info.get("profitMargins")
    growth = info.get("revenueGrowth")
    scores = {
        "ROE": score_range(roe, .05, .30),
        "Debt discipline": 100-score_range(de, 50, 200) if de is not None else 50,
        "Profit margin": score_range(margin, .05, .30),
        "Revenue growth": score_range(growth, 0, .30),
    }
    return np.mean(list(scores.values())), scores

def valuation_score(info):
    pe = info.get("trailingPE")
    peg = info.get("pegRatio")
    s1 = 50 if pe is None else 100-score_range(pe, 10, 60)
    s2 = 50 if peg is None else 100-score_range(peg, .5, 3)
    return np.mean([s1,s2]), pe, peg

if analyse:
    try:
        ticker, hist, info = get_data(stock)
        if hist.empty:
            st.error("No market data returned. Try the NSE symbol, e.g. TCS, RELIANCE, BEL.")
            st.stop()

        t = technicals(hist)
        latest = t.iloc[-1]
        price = float(latest["Close"])
        fscore, fdetails = fundamental_score(info)
        vscore, pe, peg = valuation_score(info)

        trend = 0
        if price > latest["SMA20"]: trend += 25
        if price > latest["SMA50"]: trend += 25
        if price > latest["SMA200"]: trend += 30
        if latest["RSI14"] >= 50: trend += 20
        technical_score = min(trend,100)

        overall = 0.45*fscore + 0.25*vscore + 0.30*technical_score

        if overall >= 75: verdict = "🟢 BUY / ACCUMULATE"
        elif overall >= 60: verdict = "🟡 WATCH / SELECTIVE ENTRY"
        elif overall >= 45: verdict = "🟠 HOLD / WAIT"
        else: verdict = "🔴 AVOID / HIGH RISK"

        st.subheader(f"{info.get('longName', stock)} ({stock})")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Overall Score", f"{overall:.0f}/100")
        c2.metric("Fundamental", f"{fscore:.0f}/100")
        c3.metric("Valuation", f"{vscore:.0f}/100")
        c4.metric("Technical", f"{technical_score:.0f}/100")
        st.success(verdict)

        tabs = st.tabs(["📌 Decision", "💰 Fundamentals", "📈 Technicals", "🎯 Risk & Position", "📋 Data"])

        with tabs[0]:
            a,b,c = st.columns(3)
            a.metric("Current Price", f"₹{price:,.2f}")
            b.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', np.nan):,.2f}")
            c.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', np.nan):,.2f}")
            st.markdown("### Initial thesis")
            st.write(
                "V1 combines available free market/fundamental fields with trend, momentum and valuation. "
                "Management commentary, guidance, filings and forensic checks are reserved for the next research layer."
            )

        with tabs[1]:
            st.dataframe(pd.DataFrame({
                "Metric": ["Market Cap","Revenue Growth","Profit Margin","ROE","Debt/Equity","P/E","PEG"],
                "Value": [
                    info.get("marketCap"), info.get("revenueGrowth"), info.get("profitMargins"),
                    info.get("returnOnEquity"), info.get("debtToEquity"), pe, peg
                ]
            }), use_container_width=True)
            st.write("Fundamental component scores:", fdetails)

        with tabs[2]:
            st.line_chart(t[["Close","SMA20","SMA50","SMA200"]].tail(250))
            a,b,c,d = st.columns(4)
            a.metric("RSI(14)", f"{latest['RSI14']:.1f}")
            b.metric("20 DMA", f"₹{latest['SMA20']:,.2f}")
            c.metric("50 DMA", f"₹{latest['SMA50']:,.2f}")
            d.metric("200 DMA", f"₹{latest['SMA200']:,.2f}")

        with tabs[3]:
            atr = latest["ATR14"]
            stop = price - 1.5*atr if pd.notna(atr) else price*0.95
            risk_per_share = max(price-stop, 0.01)
            risk_amount = capital*risk_pct/100
            qty = int(risk_amount/risk_per_share)
            deployed = qty*price
            st.write(f"Illustrative ATR stop: **₹{stop:,.2f}**")
            st.write(f"Risk budget: **₹{risk_amount:,.0f}**")
            st.write(f"Position size by risk: **{qty:,} shares**")
            st.write(f"Capital deployed: **₹{deployed:,.0f}**")
            st.caption("This is a mechanical illustration, not a recommendation. Slippage, liquidity and gap risk are not included.")

        with tabs[4]:
            st.dataframe(t.tail(100), use_container_width=True)

        st.info(
            "Next V1.x upgrade: NSE/BSE filing retrieval, quarterly/annual financial history, "
            "management commentary extraction, guidance-vs-actual tracker, promoter/institutional "
            "changes, peer comparison, DCF, red-flag engine and separate Long/Swing/Intraday scores."
        )

    except Exception as e:
        st.error(f"Could not analyse {stock}. Error: {e}")
else:
    st.info("Enter an NSE symbol and click ANALYSE STOCK.")
