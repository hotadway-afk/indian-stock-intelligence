import io
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import hashlib
from io import BytesIO
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(
    page_title="Indian Stock Intelligence V3.2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# V3.0: Exchange-first market-data architecture
# NSE + BSE + NSE Emerge + BSE SME security resolver
# Market-data priority: NSE/BSE exchange data -> optional yfinance fallback.
# Coverage targets:
#   - NSE Main Board
#   - NSE Emerge / SME
#   - BSE Main Board / all active equity groups
#   - BSE SME (M, MT, MS, TS groups)
#
# Important: exchange universe discovery is separated from the
# market-data provider. A stock can be in the official universe
# even when Yahoo Finance does not carry a usable history.
# ============================================================

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_SME_URL = "https://nsearchives.nseindia.com/content/equities/SME_EQUITY_L.csv"
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api"
BSE_CHART_API = "https://charting.bseindia.com/charting/RestDataProvider.svc/getDat"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
}
BSE_SME_GROUPS = {"M", "MT", "MS", "TS"}

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
    if high == low:
        return 50.0
    s = float(np.clip((x-low)/(high-low)*100, 0, 100))
    return 100-s if inverse else s


def normalize_statement(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy().replace([np.inf, -np.inf], np.nan)
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


def find_row(df, candidates):
    if df is None or df.empty:
        return None

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    rows = list(df.index)
    normalized = {norm(r): r for r in rows}

    # Exact normalized match.
    for c in candidates:
        nc = norm(c)
        if nc in normalized:
            return normalized[nc]

    # Flexible contains match, but avoid ambiguous ultra-short candidates.
    for r in rows:
        nr = norm(r)
        for c in candidates:
            nc = norm(c)
            if len(nc) >= 6 and (nc in nr or nr in nc):
                return r

    return None


# -----------------------------
# Official exchange universes
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_universe():
    frames=[]
    sources=[(NSE_EQUITY_URL,"NSE Main Board"),(NSE_SME_URL,"NSE Emerge / SME")]
    for url,segment in sources:
        try:
            r=requests.get(url,headers=REQUEST_HEADERS,timeout=30); r.raise_for_status()
            df=pd.read_csv(io.BytesIO(r.content)); df.columns=[str(c).strip().upper() for c in df.columns]
            if "SYMBOL" not in df.columns: continue
            if "SERIES" in df.columns: df=df[df["SERIES"].astype(str).str.upper().isin(["EQ","BE","SM"])].copy()
            def col(*names):
                for n in names:
                    if n in df.columns: return df[n]
                return pd.Series([""]*len(df),index=df.index)
            out=pd.DataFrame(index=df.index); out["exchange"]="NSE"; out["segment"]=segment
            out["symbol"]=col("SYMBOL").astype(str).str.strip().str.upper()
            out["company"]=col("NAME OF COMPANY","NAME_OF_COMPANY").astype(str).str.strip()
            out["isin"]=col("ISIN NUMBER","ISIN").astype(str).str.strip().str.upper()
            out["series"]=col("SERIES").astype(str).str.strip().str.upper(); out["bse_code"]=""
            out=out[out["symbol"].notna()&(out["symbol"]!="")].copy(); out["source"]="Official NSE security file"
            frames.append(out)
        except Exception:
            continue
    if not frames: return pd.DataFrame(columns=["exchange","segment","symbol","company","isin","series","bse_code","source"])
    return pd.concat(frames,ignore_index=True).drop_duplicates(subset=["exchange","segment","symbol","isin"])

BSE_VALID_GROUPS=("A","B","E","F","FC","GC","I","IF","IP","M","MS","MT","P","R","T","TS","W","X","XD","XT","Y","Z","ZP","ZY")
BSE_SME_GROUPS={"M","MS","MT","TS"}

def _pick_bse_column(df,names,default=""):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series([default]*len(df),index=df.index)

def _normalise_bse_rows(rows,group):
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    code=_pick_bse_column(df,["SCRIP_CD","ScripCode","scripcode","Security Code","scrip_id","Scrip_ID"])
    symbol=_pick_bse_column(df,["SecurityId","SCRIP_ID","Scrip_Id","SYMBOL","Symbol","Scrip ID on BOLT System"])
    name=_pick_bse_column(df,["Scrip_Name","Scrip Name","Security Name","NAME","CompanyName","Company Name"])
    isin=_pick_bse_column(df,["ISIN","ISIN_CODE","ISIN Code"])
    grp=_pick_bse_column(df,["GROUP","Group","GroupName"],group)
    out=pd.DataFrame(index=df.index); out["exchange"]="BSE"; out["group"]=grp.astype(str).str.strip().str.upper()
    out["bse_code"]=code.astype(str).str.extract(r"(\d{5,6})",expand=False).fillna("").str.strip()
    out["symbol"]=symbol.astype(str).str.strip().str.upper(); out["company"]=name.astype(str).str.strip(); out["isin"]=isin.astype(str).str.strip().str.upper(); out["series"]=out["group"]
    out["segment"]=np.where(out["group"].isin(BSE_SME_GROUPS),"BSE SME","BSE Main Board")
    bad=out["symbol"].isin(["","NAN","NONE"])|out["symbol"].str.fullmatch(r"\d+",na=False); out.loc[bad,"symbol"]=out["bse_code"]
    return out[out["bse_code"].str.len()>=5].copy()

def _bse_segment_from_group(group):
    g=str(group or "").strip().upper()
    if g in BSE_SME_GROUPS:
        return "BSE SME"
    if g:
        return "BSE Main Board"
    return "BSE / segment unverified"


def _bse_exact_from_code(code):
    """Resolve a BSE scrip directly, without requiring the bulk BSE universe to load."""
    code=str(code or "").strip()
    if not re.fullmatch(r"\d{5,6}", code):
        return pd.DataFrame()
    # Try the public BSE scrip endpoint first.
    try:
        p={"scripcode":code,"Group":"","industry":"","segment":"Equity","status":"Active"}
        r=requests.get(f"{BSE_API}/ListofScripData/w",params=p,headers=REQUEST_HEADERS,timeout=15)
        if r.ok:
            payload=r.json(); rows=payload.get("Table",payload if isinstance(payload,list) else [])
            n=_normalise_bse_rows(rows,"")
            if not n.empty:
                return n
    except Exception:
        pass
    # Then use the quote/header endpoint, which can identify a valid scrip even
    # when the bulk list endpoint is unavailable.
    try:
        r=requests.get(f"{BSE_API}/getScripHeaderData/w",params={"scripcode":code},headers=REQUEST_HEADERS,timeout=15)
        if r.ok:
            payload=r.json() if r.text else {}
            h=payload.get("Header",{}) if isinstance(payload,dict) else {}
            if isinstance(h,dict):
                company=h.get("Scrip_Name") or h.get("ScripName") or h.get("CompanyName") or ""
                symbol=h.get("ScripId") or h.get("Scrip_ID") or h.get("SecurityId") or ""
                isin=h.get("ISIN") or h.get("ISIN_CODE") or ""
                group=h.get("Group") or h.get("Scrip_Group") or h.get("GroupName") or ""
                if company or symbol or isin:
                    return pd.DataFrame([{
                        "exchange":"BSE",
                        "segment":_bse_segment_from_group(group),
                        "symbol":str(symbol).strip().upper() or code,
                        "company":str(company).strip(),
                        "isin":str(isin).strip().upper(),
                        "series":str(group).strip().upper(),
                        "bse_code":code,
                        "source":"BSE direct scrip/header resolver",
                    }])
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_bse_universe():
    frames=[]
    for group in BSE_VALID_GROUPS:
        try:
            params={"scripcode":"","Group":group,"industry":"","segment":"Equity","status":"Active"}
            r=requests.get(f"{BSE_API}/ListofScripData/w",params=params,headers=REQUEST_HEADERS,timeout=25); r.raise_for_status()
            payload=r.json(); rows=payload.get("Table",payload if isinstance(payload,list) else [])
            n=_normalise_bse_rows(rows,group)
            if not n.empty: frames.append(n)
        except Exception: continue
    if not frames: return pd.DataFrame(columns=["exchange","segment","symbol","company","isin","series","bse_code","group","source"])
    out=pd.concat(frames,ignore_index=True); out["source"]="Official BSE security-group endpoint"
    out["quality"]=(out["symbol"].astype(str).str.len()>0).astype(int)+(out["company"].astype(str).str.len()>0).astype(int)+(out["isin"].astype(str).str.len()>0).astype(int)
    out=out.sort_values("quality",ascending=False).drop_duplicates(subset=["bse_code"],keep="first").drop(columns=["quality"],errors="ignore")
    return out

@st.cache_data(ttl=900, show_spinner=False)
def bse_peer_lookup(query):
    """Best-effort BSE identity resolver independent of the bulk universe."""
    q=str(query).strip().upper()
    if not q: return pd.DataFrame()
    # Numeric BSE code: direct resolution has priority.
    if re.fullmatch(r"\d{5,6}", q):
        direct=_bse_exact_from_code(q)
        if not direct.empty: return direct
    try:
        r=requests.get(f"{BSE_API}/PeerSmartSearch/w",params={"Type":"SS","text":q},headers=REQUEST_HEADERS,timeout=15)
        r.raise_for_status()
        html=r.text.replace("&nbsp;"," ")
        codes=re.findall(r"(?<!\d)(\d{5,6})(?!\d)",html)
        isins=re.findall(r"\b(IN[A-Z0-9]{10,14})\b",html)
        code=codes[0] if codes else ""
        isin=isins[0].upper() if isins else ""
        if not code and not isin: return pd.DataFrame()
        # Enrich with exact scrip data whenever a code was found.
        if code:
            direct=_bse_exact_from_code(code)
            if not direct.empty:
                row=direct.iloc[0].to_dict()
                if isin and not row.get("isin"): row["isin"]=isin
                return pd.DataFrame([row])
        symbol=""
        for pattern in [r"<strong>\s*([A-Z0-9.&_-]{2,30})\s*</strong>",r"<b>\s*([A-Z0-9.&_-]{2,30})\s*</b>"]:
            m=re.search(pattern,html,re.I)
            if m:
                symbol=m.group(1).strip().upper(); break
        return pd.DataFrame([{
            "exchange":"BSE",
            "segment":"BSE / segment unverified",
            "symbol":symbol or q,
            "company":q,
            "isin":isin,
            "series":"",
            "bse_code":code,
            "source":"BSE PeerSmartSearch identity fallback",
        }])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    nse=load_nse_universe(); bse=load_bse_universe(); cols=["exchange","segment","symbol","company","isin","series","bse_code","source"]
    nse=nse.reindex(columns=cols); bse=bse.reindex(columns=cols); u=pd.concat([nse,bse],ignore_index=True)
    for c in cols: u[c]=u[c].fillna("").astype(str).str.strip()
    u["search_symbol"]=u["symbol"].str.upper(); u["search_company"]=u["company"].str.upper(); u["search_isin"]=u["isin"].str.upper(); u["search_bse"]=u["bse_code"].str.upper()
    return u.drop_duplicates(subset=["exchange","segment","symbol","isin","bse_code"],keep="first")

def resolve_security(query,exchange_choice,universe):
    q=str(query).strip().upper()
    if not q: return None,pd.DataFrame()
    u=universe.copy()
    if exchange_choice!="All Exchanges":
        if exchange_choice=="NSE": u=u[u.exchange=="NSE"]
        elif exchange_choice=="BSE": u=u[u.exchange=="BSE"]
        elif exchange_choice=="NSE Emerge / SME": u=u[u.segment=="NSE Emerge / SME"]
        elif exchange_choice=="BSE SME": u=u[u.segment=="BSE SME"]
    for col in ["search_symbol","search_bse","search_isin","search_company"]:
        exact=u[u[col]==q]
        if not exact.empty: return exact.iloc[0].to_dict(),exact.head(10)
    mask=(u.search_symbol.str.contains(q,na=False,regex=False)|u.search_bse.str.contains(q,na=False,regex=False)|u.search_isin.str.contains(q,na=False,regex=False)|u.search_company.str.contains(q,na=False,regex=False))
    matches=u[mask].copy()
    if not matches.empty:
        matches["_rank"]=(matches.search_symbol==q).astype(int)*100+(matches.search_symbol.str.startswith(q,na=False)).astype(int)*20+(matches.search_company.str.startswith(q,na=False)).astype(int)*10+(matches.search_bse==q).astype(int)*100+(matches.search_isin==q).astype(int)*100
        matches=matches.sort_values("_rank",ascending=False).drop(columns="_rank")
        return matches.iloc[0].to_dict(),matches.head(25)

    # Critical V3.0 change: direct BSE resolution is attempted even when the
    # bulk BSE universe is empty. This prevents provider/master failures from
    # being misreported as "unlisted".
    if exchange_choice in ("All Exchanges","BSE","BSE SME"):
        fb=bse_peer_lookup(q)
        if not fb.empty:
            row=fb.iloc[0].to_dict()
            if exchange_choice=="BSE SME" and row.get("segment") not in ("BSE SME", "BSE / segment unverified"):
                # Do not reject a valid BSE identity merely because the segment
                # could not be classified by the public endpoint.
                row["segment_note"]="BSE SME requested; group/segment could not be independently verified from the public resolver."
            return row,fb
    return None,pd.DataFrame()


def yahoo_ticker_for(sec):
    """Legacy fallback ticker only. Exchange data is preferred in V3.0."""
    exchange = str(sec.get("exchange", "NSE")).upper()
    symbol = str(sec.get("symbol", "")).strip().upper()
    code = str(sec.get("bse_code", "")).strip()
    if exchange == "BSE":
        return f"{code}.BO" if code and code.upper() not in {"NAN", "NONE"} else f"{symbol}.BO"
    return f"{symbol}.NS"


NSE_HOME = "https://www.nseindia.com"
NSE_HIST_ENDPOINT = f"{NSE_HOME}/api/historical/cm/equity"


def _nse_session():
    s = requests.Session()
    s.headers.update({
        **REQUEST_HEADERS,
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
        "Connection": "keep-alive",
    })
    return s


@st.cache_data(ttl=900, show_spinner=False)
def nse_quote(symbol):
    """Best-effort current NSE quote from the public NSE quote API."""
    try:
        s = _nse_session()
        s.get(NSE_HOME, timeout=10)
        r = s.get(f"{NSE_HOME}/api/quote-equity", params={"symbol": symbol}, timeout=15)
        r.raise_for_status()
        j = r.json()
        p = j.get("priceInfo", {}) if isinstance(j, dict) else {}
        return {
            "lastPrice": safe_num(p.get("lastPrice")),
            "previousClose": safe_num(p.get("previousClose")),
            "open": safe_num(p.get("open")),
            "dayHigh": safe_num(p.get("intraDayHighLow", {}).get("max")),
            "dayLow": safe_num(p.get("intraDayHighLow", {}).get("min")),
        }
    except Exception:
        return {}


def _nse_history_request(symbol, series, start_date, end_date):
    """Fetch one NSE historical window. NSE commonly limits large windows, so caller batches them."""
    try:
        s = _nse_session()
        s.get(NSE_HOME, timeout=10)
        params = {
            "symbol": symbol,
            "series": f'["{series}"]',
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        }
        r = s.get(NSE_HIST_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        rows = j.get("data", []) if isinstance(j, dict) else []
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def nse_history(symbol, series="EQ", years=5):
    """Exchange-first NSE OHLCV history. Batches requests to reduce public-endpoint failures."""
    symbol = str(symbol).strip().upper()
    if not symbol:
        return pd.DataFrame(), "NSE public API"
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.DateOffset(years=years)
    windows = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + pd.DateOffset(months=12), end)
        windows.append((cursor, nxt))
        cursor = nxt + pd.Timedelta(days=1)
    frames = []
    for a, b in windows:
        df = _nse_history_request(symbol, series or "EQ", a, b)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), "NSE public API"
    raw = pd.concat(frames, ignore_index=True).drop_duplicates()
    mapping = {
        "CH_TIMESTAMP": "Date",
        "CH_OPENING_PRICE": "Open",
        "CH_TRADE_HIGH_PRICE": "High",
        "CH_TRADE_LOW_PRICE": "Low",
        "CH_CLOSING_PRICE": "Close",
        "CH_LAST_TRADED_PRICE": "Last",
        "CH_PREVIOUS_CLS_PRICE": "Previous Close",
        "CH_TOT_TRADED_QTY": "Volume",
        "CH_TOT_TRADED_VAL": "Turnover",
    }
    out = pd.DataFrame()
    for src_col, dst_col in mapping.items():
        if src_col in raw.columns:
            out[dst_col] = raw[src_col]
    if "Date" not in out.columns or "Close" not in out.columns:
        return pd.DataFrame(), "NSE public API"
    out.index = pd.to_datetime(out.pop("Date"), errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Last", "Previous Close", "Volume", "Turnover"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Adj Close"] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0
    return out.sort_index().dropna(subset=["Close"]), "NSE public API"


@st.cache_data(ttl=900, show_spinner=False)
def bse_quote(code):
    try:
        r = requests.get(f"{BSE_API}/getScripHeaderData/w", params={"scripcode": code}, headers=REQUEST_HEADERS, timeout=10)
        r.raise_for_status()
        h = r.json().get("Header", {})
        out = {k: safe_num(h.get(k)) for k in ["PrevClose", "Open", "High", "Low", "LTP"]}
        for k in ["Scrip_Name", "ScripName", "CompanyName", "ScripId", "SecurityId", "ISIN", "Group", "Scrip_Group", "GroupName"]:
            if h.get(k): out[k] = h.get(k)
        return out
    except Exception:
        return {}


def _parse_bse_chart_date(value):
    """Parse BSE chart date strings used by the public charting service."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    for fmt in ["%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p", "%m/%d/%Y", "%d/%m/%Y"]:
        try:
            return pd.to_datetime(datetime.strptime(text, fmt))
        except Exception:
            pass
    return pd.to_datetime(text, errors="coerce")


def _bse_chart_history(code):
    """Fetch BSE daily OHLCV from BSE's charting service.

    This is intentionally separate from the T12M endpoint. BSE's charting
    service returns OHLCV arrays and is suitable for technical indicators.
    """
    code=str(code or "").strip()
    if not re.fullmatch(r"\d{5,6}", code):
        return pd.DataFrame()
    try:
        params={
            "exch":"B",
            "type":"b",
            "mode":"bseL",
            "fromdate":"01-01-1991-01:01:00-AM",
            "scode":code,
        }
        h={**REQUEST_HEADERS,"Origin":"https://www.bseindia.com","Accept":"application/json, text/plain, */*"}
        r=requests.post(BSE_CHART_API,params=params,headers=h,data="",timeout=30)
        if not r.ok or not r.text.strip():
            return pd.DataFrame()
        outer=r.json()
        inner=outer.get("getDatResult") if isinstance(outer,dict) else None
        if isinstance(inner,str):
            try: inner=json.loads(inner)
            except Exception: return pd.DataFrame()
        if not isinstance(inner,dict):
            inner=outer if isinstance(outer,dict) else {}

        # Some BSE responses expose TradingView-like arrays directly.
        if all(k in inner for k in ("t","o","h","l","c","v")):
            out=pd.DataFrame({
                "Date":pd.to_datetime(inner["t"],unit="s",errors="coerce"),
                "Open":pd.to_numeric(inner["o"],errors="coerce"),
                "High":pd.to_numeric(inner["h"],errors="coerce"),
                "Low":pd.to_numeric(inner["l"],errors="coerce"),
                "Close":pd.to_numeric(inner["c"],errors="coerce"),
                "Volume":pd.to_numeric(inner["v"],errors="coerce"),
            })
        else:
            divs=inner.get("DataInputValues",[])
            if not divs: return pd.DataFrame()
            d=divs[0] if isinstance(divs,list) else divs
            def arr(name,key):
                a=d.get(name,[]) if isinstance(d,dict) else []
                vals=[]
                for x in a:
                    if isinstance(x,dict): vals.append(x.get(key))
                    else: vals.append(x)
                return vals
            dates=arr("DateData","Date")
            opens=arr("OpenData","Open")
            highs=arr("HighData","High")
            lows=arr("LowData","Low")
            closes=arr("CloseData","Close")
            vols=arr("VolumeData","Volume")
            n=min(len(dates),len(opens),len(highs),len(lows),len(closes),len(vols))
            if n==0: return pd.DataFrame()
            out=pd.DataFrame({
                "Date":[_parse_bse_chart_date(x) for x in dates[:n]],
                "Open":pd.to_numeric(opens[:n],errors="coerce"),
                "High":pd.to_numeric(highs[:n],errors="coerce"),
                "Low":pd.to_numeric(lows[:n],errors="coerce"),
                "Close":pd.to_numeric(closes[:n],errors="coerce"),
                "Volume":pd.to_numeric(vols[:n],errors="coerce"),
            })
        out=out.dropna(subset=["Date","Close"]).set_index("Date").sort_index()
        if out.empty: return pd.DataFrame()
        out["Adj Close"]=out["Close"]
        return out[["Open","High","Low","Close","Adj Close","Volume"]]
    except Exception:
        return pd.DataFrame()


def bse_history_fallback(code):
    """Best-effort BSE history: charting OHLCV first, T12M second."""
    chart=_bse_chart_history(code)
    if not chart.empty:
        return chart
    try:
        r = requests.get(
            f"{BSE_API}/EquityPriceVolumeT12M/w",
            params={"scripcode": code},
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        if not r.ok:
            return pd.DataFrame()
        payload = r.json()
        data = payload.get("Data", {})
        rows = data.get("data", []) if isinstance(data, dict) else []
        if not rows:
            return pd.DataFrame()
        fields = data.get("fields", ["dttm", "vale1", "vole"])
        df = pd.DataFrame(rows, columns=fields[:len(rows[0])])
        date_col, price_col = fields[0], fields[1]
        vol_col = fields[2] if len(fields) > 2 else None
        out = pd.DataFrame(index=pd.to_datetime(df[date_col], errors="coerce"))
        out["Close"] = pd.to_numeric(df[price_col], errors="coerce")
        out["Open"] = out["Close"]
        out["High"] = out["Close"]
        out["Low"] = out["Close"]
        out["Adj Close"] = out["Close"]
        out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0
        return out.dropna(subset=["Close"]).sort_index()
    except Exception:
        return pd.DataFrame()

def _clean_yf_history(hist):
    if hist is None or hist.empty:
        return pd.DataFrame()
    h = hist.copy()
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in h.columns:
            return pd.DataFrame()
        h[c] = pd.to_numeric(h[c], errors="coerce")
    if "Adj Close" not in h.columns:
        h["Adj Close"] = h["Close"]
    return h.dropna(subset=["Close"]).sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(sec):
    """V3.0 exchange-first data layer.

    Priority for OHLCV:
      1) NSE public historical API for NSE / NSE Emerge
      2) BSE public 12M history for BSE / BSE SME
      3) yfinance only as a legacy fallback

    Fundamentals/news remain best-effort yfinance until a licensed filing/data API is connected.
    """
    exchange = str(sec.get("exchange", "")).upper()
    symbol = str(sec.get("symbol", "")).strip().upper()
    code = str(sec.get("bse_code", "")).strip()
    series = str(sec.get("series", "EQ")).strip().upper() or "EQ"
    if series not in {"EQ", "BE", "SM"}:
        series = "SM" if "SME" in str(sec.get("segment", "")) else "EQ"

    hist = pd.DataFrame()
    provider = "Unavailable"
    history_period = "—"
    provider_chain = []

    if exchange == "NSE":
        hist, provider = nse_history(symbol, series=series, years=5)
        provider_chain.append("NSE public API")
    elif exchange == "BSE" and code and code.upper() not in {"NAN", "NONE"}:
        hist = bse_history_fallback(code)
        provider = "BSE public charting API"
        history_period = "available BSE history"
        provider_chain.append("BSE charting API → BSE T12M")

    # Legacy fallback only after exchange data has failed.
    ticker_symbol = yahoo_ticker_for(sec)
    ticker = yf.Ticker(ticker_symbol)
    if hist.empty:
        try:
            hist = _clean_yf_history(ticker.history(period="5y", auto_adjust=False))
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty:
            provider = "yfinance legacy fallback"
            history_period = "5y"
        provider_chain.append("yfinance legacy fallback")
    elif history_period == "—":
        history_period = "5y"

    # Fundamentals are fetched independently so one failed yfinance endpoint
    # does not erase the other statement families.
    info = _safe_ticker_info(ticker)
    annual_income = _safe_stmt(ticker, "get_income_stmt", "yearly")
    quarterly_income = _safe_stmt(ticker, "get_income_stmt", "quarterly")
    annual_balance = _safe_stmt(ticker, "get_balance_sheet", "yearly")
    quarterly_balance = _safe_stmt(ticker, "get_balance_sheet", "quarterly")
    annual_cashflow = _safe_stmt(ticker, "get_cash_flow", "yearly")
    quarterly_cashflow = _safe_stmt(ticker, "get_cash_flow", "quarterly")

    # Exchange quote is used for the current price whenever available.
    info = _merge_quote_into_info(info, sec)

    # Derive shares from statement rows when provider metadata is absent.
    shares_from_stmt = first_available([
        statement_latest(annual_income, ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"]),
        statement_latest(quarterly_income, ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"]),
    ])
    if pd.isna(safe_num(info.get("sharesOutstanding"))) and not pd.isna(shares_from_stmt):
        info["sharesOutstanding"] = shares_from_stmt
    try:
        raw_news = ticker.news or []
        for item in raw_news[:20]:
            content = item.get("content", item)
            if isinstance(content, dict):
                title = content.get("title") or item.get("title")
                publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
                url = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link")
                pub = content.get("pubDate") or item.get("providerPublishTime")
            else:
                title = item.get("title"); publisher = item.get("publisher"); url = item.get("link"); pub = item.get("providerPublishTime")
            news.append({"title": title, "publisher": publisher, "url": url, "published": pub})
    except Exception:
        news = []

    # Current quote from the exchange should override third-party price metadata.
    if exchange == "NSE":
        nq = nse_quote(symbol)
        if nq.get("lastPrice") is not None and not pd.isna(nq.get("lastPrice")):
            info["currentPrice"] = nq.get("lastPrice")
            info["regularMarketPrice"] = nq.get("lastPrice")
    elif exchange == "BSE" and code and code.upper() not in {"NAN", "NONE"}:
        bq = bse_quote(code)
        if bq.get("LTP") is not None and not pd.isna(bq.get("LTP")):
            info["currentPrice"] = bq.get("LTP")
            info["regularMarketPrice"] = bq.get("LTP")
        if not info.get("longName"):
            info["longName"] = bq.get("Scrip_Name") or bq.get("ScripName") or bq.get("CompanyName")
        if not sec.get("isin") and bq.get("ISIN"):
            sec["isin"] = str(bq["ISIN"]).strip().upper()

    info.setdefault("longName", sec.get("company") or sec.get("symbol"))
    info.setdefault("sector", None)
    info.setdefault("industry", None)
    if sec.get("isin"):
        info["exchangeISIN"] = sec.get("isin")
    info["exchange"] = sec.get("exchange")
    info["listingSegment"] = sec.get("segment")
    info["providerTicker"] = ticker_symbol
    info["dataProvider"] = provider
    info["providerChain"] = " → ".join(provider_chain)
    info["historyPeriod"] = history_period
    info["dataQuality"] = "HIGH" if provider in {"NSE public API", "BSE public data"} and len(hist) >= 200 else ("MEDIUM" if not hist.empty else "LOW")

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
        "security": sec,
        "provider": provider,
        "history_period": history_period,
        "provider_chain": provider_chain,
    }


# -----------------------------
# Technical engine
# -----------------------------
def technical_engine(hist):
    h = hist.copy()
    if h.empty or len(h) < 30:
        return h, 0.0, {"Technical history": 0}
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
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    h["ATR14"] = tr.rolling(14).mean()
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    h["BBMid"] = mid
    h["BBUpper"] = mid + 2*std
    h["BBLower"] = mid - 2*std
    h["Vol20"] = volume.rolling(20).mean()
    h["VolumeRatio"] = volume / h["Vol20"].replace(0, np.nan)
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
    components = {
        "Price > 20DMA": 15 if price > safe_num(latest["SMA20"]) else 0,
        "Price > 50DMA": 15 if price > safe_num(latest["SMA50"]) else 0,
        "Price > 200DMA": 20 if price > safe_num(latest["SMA200"]) else 0,
        "50DMA > 200DMA": 15 if safe_num(latest["SMA50"]) > safe_num(latest["SMA200"]) else 0,
        "MACD bullish": 10 if safe_num(latest["MACD"]) > safe_num(latest["MACDSignal"]) else 0,
        "RSI healthy": 10 if 50 <= safe_num(latest["RSI14"]) <= 70 else (5 if 40 <= safe_num(latest["RSI14"]) < 50 else 0),
        "ADX > 20": 10 if safe_num(latest["ADX14"]) > 20 else 0,
        "Volume expansion": 5 if safe_num(latest["VolumeRatio"]) > 1.2 else 0,
    }
    return h, sum(components.values()), components



# -----------------------------
# V3.2 robust financial helpers
# -----------------------------

def _period_columns(df):
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        try:
            cols.append((pd.to_datetime(c), c))
        except Exception:
            pass
    return sorted(cols, key=lambda x: x[0], reverse=True)


def statement_latest(df, candidates):
    row = find_row(df, candidates)
    if row is None:
        return np.nan
    try:
        s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[0])
    except Exception:
        pass
    return np.nan


def statement_previous(df, candidates):
    row = find_row(df, candidates)
    if row is None:
        return np.nan
    try:
        s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
        if len(s) >= 2:
            return float(s.iloc[1])
    except Exception:
        pass
    return np.nan


def statement_ttm(df, candidates, n=4):
    row = find_row(df, candidates)
    if row is None:
        return np.nan
    try:
        s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
        if len(s) >= n:
            return float(s.iloc[:n].sum())
        if len(s) > 0:
            return float(s.sum())
    except Exception:
        pass
    return np.nan


def statement_ttm_previous(df, candidates, n=4):
    row = find_row(df, candidates)
    if row is None:
        return np.nan
    try:
        s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
        if len(s) >= 2*n:
            return float(s.iloc[n:2*n].sum())
    except Exception:
        pass
    return np.nan


def first_available(values):
    for v in values:
        if not pd.isna(v):
            return v
    return np.nan


def pct_change(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b - 1


def get_price(info, hist):
    p = safe_num(info.get("currentPrice"))
    if not pd.isna(p) and p > 0:
        return p
    if hist is not None and not hist.empty and "Close" in hist:
        try:
            return float(pd.to_numeric(hist["Close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass
    return np.nan


def get_shares(info, annual_income=None, annual_balance=None, quarterly_income=None):
    for key in [
        "sharesOutstanding", "impliedSharesOutstanding",
        "floatShares", "sharesIssued", "shares"
    ]:
        v = safe_num(info.get(key))
        if not pd.isna(v) and v > 0:
            return v

    for df in [annual_income, quarterly_income, annual_balance]:
        row = find_row(df, [
            "Ordinary Shares Number", "Common Stock Shares Outstanding",
            "Share Issued", "Diluted Average Shares", "Basic Average Shares",
            "Common Stock Shares Outstanding"
        ])
        if row is not None:
            try:
                s = pd.to_numeric(df.loc[row], errors="coerce").dropna()
                if not s.empty and s.iloc[0] > 0:
                    return float(s.iloc[0])
            except Exception:
                pass
    return np.nan


def data_availability_score(fundamental, valuation, technical_score, management):
    checks = {
        "Revenue": not pd.isna(fundamental.get("revenue")),
        "PAT": not pd.isna(fundamental.get("net_income")),
        "ROE": not pd.isna(fundamental.get("roe")),
        "Leverage": not pd.isna(fundamental.get("de")),
        "FCF": not pd.isna(fundamental.get("fcf")),
        "Valuation": valuation.get("available_metrics", 0) >= 2,
        "Technical": not pd.isna(technical_score),
        # Missing uploaded management docs should not make an otherwise good company
        # look "data incomplete" in the same way as missing financial statements.
        "Evidence layer": management.get("score", 0) > 0,
    }
    score = round(100 * sum(checks.values()) / len(checks))
    return score, checks


# -----------------------------
# Fundamental / valuation / quality engines
# -----------------------------
def fundamental_engine(info, annual_income, annual_balance, annual_cashflow,
                       quarterly_income=None, quarterly_balance=None, quarterly_cashflow=None):
    quarterly_income = quarterly_income if quarterly_income is not None else pd.DataFrame()
    quarterly_balance = quarterly_balance if quarterly_balance is not None else pd.DataFrame()
    quarterly_cashflow = quarterly_cashflow if quarterly_cashflow is not None else pd.DataFrame()

    revenue_a = statement_latest(annual_income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    revenue_p_a = statement_previous(annual_income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    pat_a = statement_latest(annual_income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    pat_p_a = statement_previous(annual_income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    ebitda_a = statement_latest(annual_income, ["EBITDA", "Normalized EBITDA"])
    ebit_a = statement_latest(annual_income, ["EBIT", "Operating Income"])

    revenue_ttm = statement_ttm(quarterly_income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    revenue_ttm_prev = statement_ttm_previous(quarterly_income, ["Total Revenue", "Operating Revenue", "TotalRevenue"])
    pat_ttm = statement_ttm(quarterly_income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    pat_ttm_prev = statement_ttm_previous(quarterly_income, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    ebitda_ttm = statement_ttm(quarterly_income, ["EBITDA", "Normalized EBITDA"])
    ebit_ttm = statement_ttm(quarterly_income, ["EBIT", "Operating Income"])

    revenue = first_available([revenue_a, revenue_ttm])
    revenue_prev = first_available([revenue_p_a, revenue_ttm_prev])
    net_income = first_available([pat_a, pat_ttm])
    net_income_prev = first_available([pat_p_a, pat_ttm_prev])
    ebitda = first_available([ebitda_a, ebitda_ttm])
    ebit = first_available([ebit_a, ebit_ttm])

    debt = first_available([
        statement_latest(annual_balance, ["Total Debt", "Total Debt And Capital Lease Obligation", "Long Term Debt And Capital Lease Obligation", "Current Debt"]),
        statement_latest(quarterly_balance, ["Total Debt", "Total Debt And Capital Lease Obligation", "Long Term Debt And Capital Lease Obligation", "Current Debt"])
    ])
    equity = first_available([
        statement_latest(annual_balance, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "Total Equity", "Shareholders Equity"]),
        statement_latest(quarterly_balance, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "Total Equity", "Shareholders Equity"])
    ])
    cash = first_available([
        statement_latest(annual_balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"]),
        statement_latest(quarterly_balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"])
    ])
    assets = first_available([
        statement_latest(annual_balance, ["Total Assets"]),
        statement_latest(quarterly_balance, ["Total Assets"])
    ])

    operating_cf = first_available([
        statement_latest(annual_cashflow, [
            "Operating Cash Flow", "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities"
        ]),
        statement_ttm(quarterly_cashflow, [
            "Operating Cash Flow", "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities"
        ])
    ])
    capex = first_available([
        statement_latest(annual_cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment"]),
        statement_ttm(quarterly_cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment"])
    ])
    fcf = np.nan
    if not pd.isna(operating_cf) and not pd.isna(capex):
        fcf = operating_cf + capex if capex < 0 else operating_cf - capex

    roe = (net_income / equity) if not pd.isna(net_income) and not pd.isna(equity) and equity != 0 else safe_num(info.get("returnOnEquity"))
    roa = (net_income / assets) if not pd.isna(net_income) and not pd.isna(assets) and assets != 0 else safe_num(info.get("returnOnAssets"))
    margin = (net_income / revenue) if not pd.isna(net_income) and not pd.isna(revenue) and revenue != 0 else safe_num(info.get("profitMargins"))
    de = (debt / equity) if not pd.isna(debt) and not pd.isna(equity) and equity != 0 else safe_num(info.get("debtToEquity"))
    rev_growth = pct_change(revenue, revenue_prev) if not pd.isna(revenue_prev) else safe_num(info.get("revenueGrowth"))
    earnings_growth = pct_change(net_income, net_income_prev) if not pd.isna(net_income_prev) else safe_num(info.get("earningsGrowth"))
    cfo_pat = operating_cf / net_income if not pd.isna(operating_cf) and not pd.isna(net_income) and net_income != 0 else np.nan

    raw = {
        "Revenue growth": score_range(rev_growth, -0.05, 0.25),
        "Earnings growth": score_range(earnings_growth, -0.10, 0.30),
        "ROE": score_range(roe, 0.05, 0.25),
        "ROA": score_range(roa, 0.02, 0.15),
        "Profit margin": score_range(margin, 0.03, 0.25),
        "Debt discipline": score_range(de, 25, 150, inverse=True),
        "CFO / PAT": score_range(cfo_pat, 0.60, 1.20),
    }
    usable = {k: v for k, v in raw.items() if not pd.isna(v)}
    score = float(np.mean(list(usable.values()))) if usable else 50.0

    return {
        "score": score,
        "components": usable if usable else {"No fundamental data": 50.0},
        "revenue": revenue,
        "revenue_prev": revenue_prev,
        "net_income": net_income,
        "net_income_prev": net_income_prev,
        "ebitda": ebitda,
        "ebit": ebit,
        "operating_cf": operating_cf,
        "capex": capex,
        "fcf": fcf,
        "debt": debt,
        "equity": equity,
        "cash": cash,
        "assets": assets,
        "roe": roe,
        "roa": roa,
        "margin": margin,
        "rev_growth": rev_growth,
        "earnings_growth": earnings_growth,
        "de": de,
        "cfo_pat": cfo_pat,
        "available_metrics": len(usable),
        "annual_income": annual_income,
        "annual_balance": annual_balance,
        "quarterly_income": quarterly_income,
        "quarterly_balance": quarterly_balance,
    }


def valuation_engine(info, fundamental, hist=None):
    hist = hist if hist is not None else pd.DataFrame()

    # Current price: exchange quote first, then local OHLCV.
    price = get_price(info, hist)

    shares = get_shares(
        info,
        fundamental.get("annual_income"),
        fundamental.get("annual_balance"),
        fundamental.get("quarterly_income"),
    )

    market_cap = safe_num(info.get("marketCap"))
    if pd.isna(market_cap) and not pd.isna(price) and not pd.isna(shares):
        market_cap = price * shares

    # Additional provider fields if available.
    if pd.isna(market_cap):
        market_cap = safe_num(info.get("enterpriseValue"))

    net_income = fundamental.get("net_income")
    revenue = fundamental.get("revenue")
    equity = fundamental.get("equity")
    ebitda = fundamental.get("ebitda")
    debt = fundamental.get("debt")
    cash = fundamental.get("cash")

    pe = (
        market_cap / net_income
        if not pd.isna(market_cap) and not pd.isna(net_income) and net_income > 0
        else safe_num(info.get("trailingPE"))
    )
    pb = (
        market_cap / equity
        if not pd.isna(market_cap) and not pd.isna(equity) and equity > 0
        else safe_num(info.get("priceToBook"))
    )
    ps = (
        market_cap / revenue
        if not pd.isna(market_cap) and not pd.isna(revenue) and revenue > 0
        else safe_num(info.get("priceToSalesTrailing12Months"))
    )

    ev = np.nan
    if not pd.isna(market_cap):
        ev = market_cap + (debt if not pd.isna(debt) else 0)
        ev -= (cash if not pd.isna(cash) else 0)

    ev_ebitda = (
        ev / ebitda
        if not pd.isna(ev) and not pd.isna(ebitda) and ebitda > 0
        else safe_num(info.get("enterpriseToEbitda"))
    )

    forward_pe = safe_num(info.get("forwardPE"))
    peg = safe_num(info.get("pegRatio"))
    if pd.isna(peg) and not pd.isna(pe):
        eg = fundamental.get("earnings_growth")
        if not pd.isna(eg) and eg > 0:
            peg = pe / (eg * 100)

    metrics = {
        "P/E": pe,
        "Forward P/E": forward_pe,
        "PEG": peg,
        "P/B": pb,
        "EV/EBITDA": ev_ebitda,
        "P/S": ps,
    }

    parts = {}
    bounds = {
        "P/E": (10, 60),
        "Forward P/E": (8, 50),
        "PEG": (0.5, 3),
        "P/B": (1, 10),
        "EV/EBITDA": (5, 35),
        "P/S": (0.5, 8),
    }
    for name, value in metrics.items():
        if not pd.isna(value):
            lo, hi = bounds[name]
            parts[name] = score_range(value, lo, hi, True)

    return {
        "score": float(np.mean(list(parts.values()))) if parts else 50.0,
        "parts": parts if parts else {"No valuation data": 50.0},
        "pe": pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "ps": ps,
        "market_cap": market_cap,
        "enterprise_value": ev,
        "available_metrics": len(parts),
        "price": price,
        "shares": shares,
    }


def quality_engine(info, fundamental):
    flags, positives, checks = [], [], []

    roe = fundamental.get("roe")
    if not pd.isna(roe):
        checks.append(100 if roe >= 0.15 else 70 if roe >= 0.10 else 40 if roe >= 0.05 else 10)
        if roe >= 0.15:
            positives.append(f"ROE is {roe*100:.1f}%.")
        elif roe < 0.05:
            flags.append(f"ROE is weak at {roe*100:.1f}%.")

    rg = fundamental.get("rev_growth")
    if not pd.isna(rg):
        checks.append(100 if rg >= 0.15 else 75 if rg >= 0.05 else 45 if rg >= 0 else 15)
        if rg >= 0.15:
            positives.append(f"Revenue grew {rg*100:.1f}% YoY — strong growth.")
        elif rg < 0:
            flags.append(f"Revenue declined {abs(rg)*100:.1f}% YoY.")

    eg = fundamental.get("earnings_growth")
    if not pd.isna(eg):
        checks.append(100 if eg >= 0.15 else 75 if eg >= 0.05 else 45 if eg >= 0 else 10)
        if eg < 0:
            flags.append(f"Earnings growth is negative ({eg*100:.1f}%).")

    de = fundamental.get("de")
    if not pd.isna(de):
        checks.append(100 if de <= 0.5 else 80 if de <= 1 else 55 if de <= 2 else 20)
        if de > 2:
            flags.append(f"Debt/equity is elevated at {de:.2f}x.")

    cfo_pat = fundamental.get("cfo_pat")
    if not pd.isna(cfo_pat):
        checks.append(100 if cfo_pat >= 1.1 else 80 if cfo_pat >= 0.9 else 50 if cfo_pat >= 0.6 else 20)
        if cfo_pat < 0.6:
            flags.append(f"Cash conversion is weak at {cfo_pat:.2f}x PAT.")

    fcf = fundamental.get("fcf")
    ni = fundamental.get("net_income")
    if not pd.isna(fcf):
        if fcf > 0:
            checks.append(80)
            positives.append("Latest available free cash flow is positive.")
        else:
            checks.append(25)
            flags.append("Latest available free cash flow is negative.")
            if not pd.isna(ni) and ni > 0:
                flags.append("Positive accounting profit with negative free cash flow; investigate capex/earnings quality.")

    promoter = safe_num(info.get("heldPercentInsiders"))
    if not pd.isna(promoter):
        checks.append(100 if promoter >= 0.40 else 75 if promoter >= 0.25 else 55 if promoter >= 0.15 else 30)
        if promoter < 0.15:
            flags.append(f"Insider/promoter ownership is low at {promoter*100:.1f}% in the available snapshot.")

    score = float(np.mean(checks)) if checks else 50.0
    return max(0, min(100, score)), positives, flags



# -----------------------------
# V3.0 management / evidence engine
# -----------------------------

EVIDENCE_KEYWORDS = {
    "guidance": [
        "guidance", "outlook", "expect", "expected", "target", "FY27", "FY28",
        "revenue growth", "ebitda margin", "margin guidance"
    ],
    "capex": [
        "capex", "capital expenditure", "capacity expansion", "brownfield",
        "greenfield", "capacity addition", "new plant", "expansion"
    ],
    "order_book": [
        "order book", "orderbook", "order intake", "orders", "order inflow",
        "book-to-bill", "pipeline"
    ],
    "demand": [
        "demand", "industry outlook", "end market", "utilisation", "capacity utilisation",
        "volume growth", "volume"
    ],
    "margin": [
        "gross margin", "ebitda margin", "operating margin", "margin expansion",
        "margin improvement", "cost optimisation"
    ],
    "risk": [
        "risk", "headwind", "challenge", "uncertainty", "commodity", "forex",
        "customer concentration", "competition"
    ],
    "capital_allocation": [
        "dividend", "buyback", "deleveraging", "debt reduction", "capital allocation",
        "acquisition", "merger"
    ],
}

def extract_uploaded_documents(uploaded_files):
    docs = []
    if not uploaded_files:
        return docs
    for f in uploaded_files:
        name = f.name
        raw = f.getvalue()
        sha = hashlib.sha256(raw).hexdigest()[:12]
        text = ""
        if name.lower().endswith(".pdf"):
            if PdfReader is None:
                text = ""
            else:
                try:
                    reader = PdfReader(BytesIO(raw))
                    pages = []
                    for page in reader.pages[:80]:
                        try:
                            pages.append(page.extract_text() or "")
                        except Exception:
                            pass
                    text = "\n".join(pages)
                except Exception:
                    text = ""
        else:
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        docs.append({"name": name, "sha": sha, "text": text})
    return docs

def evidence_snippets(docs, limit_per_topic=3):
    findings = []
    for topic, keywords in EVIDENCE_KEYWORDS.items():
        count = 0
        for doc in docs:
            t = re.sub(r"\s+", " ", doc["text"])
            if not t:
                continue
            lower = t.lower()
            for kw in keywords:
                pos = lower.find(kw.lower())
                if pos >= 0:
                    left = max(0, pos - 180)
                    right = min(len(t), pos + 420)
                    snippet = t[left:right].strip()
                    findings.append({
                        "topic": topic.replace("_", " ").title(),
                        "document": doc["name"],
                        "evidence": snippet,
                        "matched": kw,
                    })
                    count += 1
                    if count >= limit_per_topic:
                        break
            if count >= limit_per_topic:
                break
    return findings

def management_evidence_engine(sec, info, docs):
    evidence = []
    if sec.get("exchange"):
        evidence.append({
            "topic": "Identity",
            "source": "Exchange security master",
            "detail": f"{sec.get('exchange')} / {sec.get('segment')} / {sec.get('symbol')}"
        })
    if sec.get("isin"):
        evidence.append({
            "topic": "Identity",
            "source": "Exchange",
            "detail": f"ISIN {sec.get('isin')}"
        })
    provider = info.get("dataProvider") or "Unavailable"
    evidence.append({
        "topic": "Market Data",
        "source": "Provider chain",
        "detail": provider
    })

    findings = evidence_snippets(docs)
    topics = {f["topic"] for f in findings}

    quality_points = 25  # exchange identity
    if info.get("exchangeISIN"):
        quality_points += 10
    if info.get("dataProvider") not in (None, "", "Unavailable"):
        quality_points += 10
    if docs:
        quality_points += 25
    if any(f["topic"] == "Guidance" for f in findings):
        quality_points += 10
    if any(f["topic"] == "Capex" for f in findings):
        quality_points += 10
    if any(f["topic"] == "Order Book" for f in findings):
        quality_points += 10
    quality_points = min(100, quality_points)

    if docs and findings:
        confidence = "MEDIUM/HIGH — company documents uploaded and searched"
    elif docs:
        confidence = "MEDIUM — documents uploaded but few evidence hits"
    else:
        confidence = "LOW — company-specific documents not provided"

    return {
        "score": float(quality_points),
        "confidence": confidence,
        "evidence": evidence,
        "findings": findings,
        "guidance": "Needs document verification",
        "note": (
            "V3.2 separates evidence availability from management quality. "
            "It will not convert missing guidance into a neutral 50/100 management score. "
            "Upload an annual report, investor presentation or earnings-call transcript "
            "to extract company-specific guidance, capex, order-book and outlook evidence."
        )
    }

def source_links(sec):
    links = []
    sym = str(sec.get("symbol") or "").strip().upper()
    code = str(sec.get("bse_code") or "").strip()
    exchange = str(sec.get("exchange") or "").upper()

    if exchange == "NSE" and sym:
        links.append(("NSE quote", f"https://www.nseindia.com/get-quotes/equity?symbol={sym}"))
        links.append(("NSE announcements", f"https://www.nseindia.com/companies-listing/corporate-filings-announcements"))
        links.append(("NSE annual reports", "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"))
    if exchange == "BSE" and code:
        links.append(("BSE quote", f"https://m.bseindia.com/new/GetQuote.aspx?scripcd={code}"))
        links.append(("BSE market-data portal", "https://marketdata.bseindia.com/"))
    links.append(("NSE all reports", "https://www.nseindia.com/all-reports"))
    links.append(("SEBI filings search", "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=78&ssid=15"))
    return links


# -----------------------------
# UI
# -----------------------------
st.title("📊 Indian Stock Intelligence — V3.2")
st.caption("Exchange-first NSE + BSE + NSE Emerge + BSE SME Fundamental + Evidence + Valuation + Technical + Risk engine")

universe = load_universe()

with st.sidebar:
    st.header("Analysis Inputs")

with st.sidebar:
    st.subheader("📄 Evidence / filing documents")
    st.caption(
        "Optional: upload an annual report, investor presentation, earnings-call transcript "
        "or other company filing. V3.0 searches the documents for guidance, capex, order book, "
        "demand, margin and risk evidence."
    )
    uploaded = st.file_uploader(
        "Upload PDF/TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="evidence_upload"
    )
    if uploaded:
        st.session_state["uploaded_docs"] = extract_uploaded_documents(uploaded)
    else:
        st.session_state.setdefault("uploaded_docs", [])
    exchange_choice = st.selectbox("Exchange / segment", ["All Exchanges", "NSE", "BSE", "NSE Emerge / SME", "BSE SME"])
    stock = st.text_input("Symbol / BSE code / ISIN / company", "RELIANCE").strip().upper()
    capital = st.number_input("Portfolio capital (₹)", min_value=10000, value=500000, step=10000)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    objective = st.selectbox("Primary objective", ["Long Term", "Swing Trading", "Intraday"])
    run = st.button("🚀 RUN V3.2 ANALYSIS", type="primary", use_container_width=True)

if not run:
    st.info("Enter a symbol, BSE code, ISIN or company name and click RUN V3.2 ANALYSIS.")
    a,b,c,d = st.columns(4)
    nse_count = len(universe[universe.exchange == "NSE"]) if not universe.empty else 0
    bse_count = len(universe[universe.exchange == "BSE"]) if not universe.empty else 0
    nse_sme = len(universe[universe.segment == "NSE Emerge / SME"]) if not universe.empty else 0
    bse_sme = len(universe[universe.segment == "BSE SME"]) if not universe.empty else 0
    a.metric("NSE universe", f"{nse_count:,}")
    b.metric("BSE universe", f"{bse_count:,}")
    c.metric("NSE Emerge / SME", f"{nse_sme:,}")
    d.metric("BSE SME", f"{bse_sme:,}")
    st.markdown("### V3.2 coverage")
    st.markdown("- NSE Main Board + NSE Emerge / SME")
    st.markdown("- BSE Main Board + BSE SME")
    st.markdown("- Symbol, BSE scrip code, ISIN and company-name lookup")
    st.markdown("- Exchange-first market data where publicly available")
    st.markdown("- Fundamentals and valuation can continue even if technical history is unavailable")
    st.markdown("- Optional annual-report / investor-presentation / transcript evidence extraction")
    st.caption(
        "V3.2 separates security identity, market-data availability and research evidence. "
        "Technical analysis is optional; research analysis can continue when OHLCV is unavailable."
    )
    st.stop()

try:
    sec, matches = resolve_security(stock, exchange_choice, universe)
    if sec is None:
        # Cross-segment verification: a valid security may simply be in another
        # exchange/segment than the one selected by the user.
        alt_sec, alt_matches = resolve_security(stock, "All Exchanges", universe)

        st.error(f"Could not resolve '{stock}' in the selected exchange/segment universe.")

        if alt_sec is not None:
            st.warning(
                f"Security found elsewhere: {alt_sec.get('exchange')} / "
                f"{alt_sec.get('segment')} / {alt_sec.get('symbol') or stock}."
            )
            st.dataframe(
                alt_matches[["exchange","segment","symbol","company","isin","bse_code","source"]]
                if "source" in alt_matches.columns
                else alt_matches[["exchange","segment","symbol","company","isin","bse_code"]],
                use_container_width=True,
                hide_index=True
            )
            st.info(
                "Change the Exchange / segment selector to the segment shown above and run the analysis again."
            )
        elif not matches.empty:
            st.dataframe(
                matches[["exchange","segment","symbol","company","isin","bse_code","source"]]
                if "source" in matches.columns
                else matches[["exchange","segment","symbol","company","isin","bse_code"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "The current public exchange identity sources did not return a match. "
                "Try the BSE code, ISIN, or company name."
            )
        st.stop()

    if len(matches) > 1 and stock not in [str(sec.get("symbol","")), str(sec.get("bse_code",""))]:
        st.info("Multiple matches found. The first exact/best match is being analysed.")
        st.dataframe(matches[["exchange","segment","symbol","company","isin","bse_code","source"]] if "source" in matches.columns else matches[["exchange","segment","symbol","company","isin","bse_code"]], use_container_width=True, hide_index=True)

    with st.spinner(f"Loading {sec.get('company') or stock} data..."):
        data = load_stock(sec)

    hist = data["hist"]; info = data["info"]

    # V3.0: technicals are optional. A company can still receive a research
    # report when OHLCV history is unavailable.
    if hist.empty:
        tech = hist
        technical_score = np.nan
        technical_components = {"Technical history": 0}
    else:
        tech, technical_score, technical_components = technical_engine(hist)

    fundamental = fundamental_engine(
        info,
        data["annual_income"],
        data["annual_balance"],
        data["annual_cashflow"],
        data.get("quarterly_income"),
        data.get("quarterly_balance"),
        data.get("quarterly_cashflow"),
    )
    valuation = valuation_engine(info, fundamental, hist)
    quality_score, positives, flags = quality_engine(info, fundamental)

    uploaded_files = st.session_state.get("uploaded_docs", [])
    management = management_evidence_engine(sec, info, uploaded_files)

    # Re-weight the score when technical data is genuinely unavailable.
    base_weights = {
        "fundamental": 0.40,
        "management": 0.20,
        "valuation": 0.20,
        "technical": 0.10,
        "quality": 0.10,
    }
    available = {
        "fundamental": not pd.isna(fundamental["score"]),
        "management": not pd.isna(management["score"]),
        "valuation": not pd.isna(valuation["score"]),
        "technical": not pd.isna(technical_score),
        "quality": not pd.isna(quality_score),
    }
    weights = {k: v for k, v in base_weights.items() if available[k]}
    total_weight = sum(weights.values()) or 1
    weights = {k: v / total_weight for k, v in weights.items()}

    overall = (
        fundamental["score"] * weights.get("fundamental", 0)
        + management["score"] * weights.get("management", 0)
        + valuation["score"] * weights.get("valuation", 0)
        + (technical_score if not pd.isna(technical_score) else 0) * weights.get("technical", 0)
        + quality_score * weights.get("quality", 0)
    )

    if objective == "Long Term":
        obj_weights = {"fundamental": 0.45, "management": 0.25, "valuation": 0.20, "quality": 0.10}
    elif objective == "Swing Trading":
        obj_weights = {"technical": 0.55, "valuation": 0.20, "quality": 0.15, "fundamental": 0.10}
    else:
        obj_weights = {"technical": 0.75, "quality": 0.15, "valuation": 0.10}

    obj_available = {k: v for k, v in obj_weights.items() if available.get(k, False)}
    obj_total = sum(obj_available.values()) or 1
    obj_weights = {k: v / obj_total for k, v in obj_available.items()}
    objective_score = (
        fundamental["score"] * obj_weights.get("fundamental", 0)
        + management["score"] * obj_weights.get("management", 0)
        + valuation["score"] * obj_weights.get("valuation", 0)
        + (technical_score if not pd.isna(technical_score) else 0) * obj_weights.get("technical", 0)
        + quality_score * obj_weights.get("quality", 0)
    )
    overall = float(np.clip(overall, 0, 100))
    objective_score = float(np.clip(objective_score, 0, 100))

    if objective_score >= 75: verdict="🟢 STRONG SETUP"
    elif objective_score >= 60: verdict="🟡 SELECTIVE / WATCH"
    elif objective_score >= 45: verdict="🟠 WAIT / NEUTRAL"
    else: verdict="🔴 AVOID / HIGH RISK"

    company = info.get("longName") or sec.get("company") or stock
    if hist.empty:
        price = safe_num(info.get("currentPrice"))
        high52 = safe_num(info.get("fiftyTwoWeekHigh"))
        low52 = safe_num(info.get("fiftyTwoWeekLow"))
    else:
        price = safe_num(tech.iloc[-1]["Close"])
        high52 = safe_num(info.get("fiftyTwoWeekHigh")); low52 = safe_num(info.get("fiftyTwoWeekLow"))
        if pd.isna(high52): high52 = safe_num(tech["High"].tail(252).max())
        if pd.isna(low52): low52 = safe_num(tech["Low"].tail(252).min())

    st.subheader(f"{company} ({sec.get('symbol') or stock})")
    st.caption(f"{sec.get('exchange')} • {sec.get('segment')} • BSE code: {sec.get('bse_code') or '—'} • ISIN: {sec.get('isin') or '—'} • Universe: {sec.get('source') or '—'} • Data: {data['provider']} ({data['history_period']})")

    with st.expander("🔎 Security identity / exchange verification", expanded=True):
        id1,id2,id3,id4=st.columns(4)
        id1.metric("Exchange",str(sec.get("exchange") or "N/A")); id2.metric("Segment",str(sec.get("segment") or "N/A")); id3.metric("BSE code",str(sec.get("bse_code") or "N/A")); id4.metric("ISIN",str(sec.get("isin") or "N/A"))
        if sec.get("segment_note"): st.warning(sec.get("segment_note"))
        st.caption("Exchange identity is authoritative. V3.2 attempts NSE/BSE exchange data first; yfinance remains a legacy fallback only. Missing provider history is never treated as proof of an unlisted security.")
        st.markdown("#### Market-data provenance")
        st.write({
            "Primary provider used": data.get("provider"),
            "Provider chain attempted": data.get("provider_chain"),
            "History available": f"{len(data.get('hist', pd.DataFrame())):,} sessions",
            "History period": data.get("history_period"),
            "Data quality": info.get("dataQuality", "LOW"),
        })

    cols=st.columns(6)
    cols[0].metric("V3.0 Score", f"{overall:.0f}/100")
    cols[1].metric("Fundamental", f"{fundamental['score']:.0f}")
    cols[2].metric("Evidence", f"{management['score']:.0f}")
    cols[3].metric("Valuation", f"{valuation['score']:.0f}")
    cols[4].metric("Technical", f"{technical_score:.0f}")
    cols[5].metric("Quality/Risk", f"{quality_score:.0f}")

    if "STRONG" in verdict: st.success(verdict)
    elif "WATCH" in verdict: st.warning(verdict)
    elif "WAIT" in verdict: st.info(verdict)
    else: st.error(verdict)
    completeness_score, completeness_checks = data_availability_score(
        fundamental, valuation, technical_score, management
    )
    st.caption(
        f"Primary objective: {objective} | Objective-specific score: {objective_score:.0f}/100 "
        f"| Data completeness: {completeness_score}/100"
    )
    if completeness_score < 50:
        st.warning(
            "Data completeness is low. The verdict is provisional; do not treat it as a strong investment signal."
        )

    tabs=st.tabs([
        "🎯 Executive Decision","🏢 Fundamentals","📊 Quarterly","🧠 Management / Evidence",
        "💰 Valuation","📈 Technicals","🚨 Risks & Catalysts","📰 Recent News",
        "📚 Data & Sources","📋 Raw Financials"
    ])

    with tabs[0]:
        a,b,c,d=st.columns(4)
        a.metric("Price", f"₹{price:,.2f}")
        b.metric("52W High", f"₹{high52:,.2f}" if not pd.isna(high52) else "N/A")
        c.metric("52W Low", f"₹{low52:,.2f}" if not pd.isna(low52) else "N/A")
        mcap=info.get("marketCap")
        d.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
        st.markdown("### Coverage & decision framework")
        st.write(f"**Decision:** {verdict}")
        st.write(f"**Evidence confidence:** {management['confidence']}")
        st.write(f"**Data completeness:** {completeness_score}/100")
        st.write(f"Exchange identity is resolved independently from market data. **OHLCV source:** {data.get('provider')} | **History:** {len(hist):,} sessions | **Quality:** {info.get('dataQuality', 'LOW')}.")
        st.markdown("### Key positives")
        for p in positives: st.success("✓ "+p)
        if not positives: st.write("No strong positive flags identified by the current structured-data rules.")
        st.markdown("### Key concerns")
        for f in flags: st.error("⚠ "+f)
        if not flags: st.success("No major automated red flags triggered.")

    with tabs[1]:
        st.markdown("### Fundamental scorecard")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "Revenue","PAT","EBITDA","ROE","ROA","Revenue growth",
                    "Earnings growth","Profit margin","Debt/Equity","CFO/PAT","FCF"
                ],
                "Value": [
                    fmt_num(fundamental["revenue"]),
                    fmt_num(fundamental["net_income"]),
                    fmt_num(fundamental["ebitda"]),
                    fmt_pct(fundamental["roe"]),
                    fmt_pct(fundamental["roa"]),
                    fmt_pct(fundamental["rev_growth"]),
                    fmt_pct(fundamental["earnings_growth"]),
                    fmt_pct(fundamental["margin"]),
                    fmt_num(fundamental["de"]),
                    fmt_num(fundamental["cfo_pat"]),
                    fmt_num(fundamental["fcf"]),
                ],
            }),
            use_container_width=True,
            hide_index=True
        )
        st.markdown(f"**Fundamental score: {fundamental['score']:.0f}/100**")
        st.bar_chart(pd.Series(fundamental["components"], dtype="float64"))
        st.markdown("### Annual income statement")
        st.dataframe(data["annual_income"],use_container_width=True)


    with tabs[2]:
        st.markdown("### Quarterly income statement")
        st.dataframe(data["quarterly_income"],use_container_width=True)

        qrev = statement_ttm(data.get("quarterly_income"), ["Total Revenue", "Operating Revenue", "TotalRevenue"])
        qrev_prev = statement_ttm_previous(data.get("quarterly_income"), ["Total Revenue", "Operating Revenue", "TotalRevenue"])
        qpat = statement_ttm(data.get("quarterly_income"), ["Net Income", "Net Income Common Stockholders", "NetIncome"])
        qpat_prev = statement_ttm_previous(data.get("quarterly_income"), ["Net Income", "Net Income Common Stockholders", "NetIncome"])

        st.markdown("### TTM trend")
        a,b,c,d = st.columns(4)
        a.metric("TTM Revenue", fmt_num(qrev))
        b.metric("TTM PAT", fmt_num(qpat))
        c.metric("TTM Revenue YoY", fmt_pct(pct_change(qrev, qrev_prev)))
        d.metric("TTM PAT YoY", fmt_pct(pct_change(qpat, qpat_prev)))

        st.markdown("### Quarterly balance sheet")
        st.dataframe(data["quarterly_balance"],use_container_width=True)
        st.markdown("### Quarterly cash flow")
        st.dataframe(data["quarterly_cashflow"],use_container_width=True)


    with tabs[3]:
        st.markdown("### Management / Evidence")
        a,b,c = st.columns(3)
        a.metric("Evidence score", f"{management['score']:.0f}/100")
        b.metric("Evidence confidence", management["confidence"].split(" — ")[0])
        c.metric("Guidance status", str(management.get("guidance", "Not verified")))

        st.info(management["note"])

        st.markdown("### Verified structured evidence")
        ev = management.get("evidence", [])
        if ev:
            for e in ev:
                if isinstance(e, dict):
                    st.write(f"• **{e.get('topic','Evidence')}** — {e.get('source','Source')}: {e.get('detail','')}")
                else:
                    st.write("• " + str(e))
        else:
            st.info("No structured evidence items were returned.")

        st.markdown("### Extracted company-document evidence")
        findings = management.get("findings", [])
        if findings:
            for f in findings[:20]:
                if isinstance(f, dict):
                    st.markdown(f"**{f.get('topic','Evidence')} — {f.get('document','Document')}**")
                    st.caption(f.get("evidence",""))
        else:
            st.warning("No company-specific documents were uploaded in this session.")

        st.markdown("### Guidance tracker")
        st.write("FY27 revenue guidance: " + str(management.get("guidance", "Not verified")))
        st.write("FY27 EBITDA / margin guidance: Not verified unless supported by company evidence.")
        st.write("FY28 revenue guidance: Not verified unless supported by company evidence.")
        st.write("FY28 EBITDA / margin guidance: Not verified unless supported by company evidence.")
        st.write("Order book / capex / capacity: Not verified unless supported by company evidence.")


    with tabs[4]:
        st.markdown("### Relative valuation")
        st.dataframe(
            pd.DataFrame({
                "Metric":["Trailing P/E","Forward P/E","PEG","P/B","EV/EBITDA","P/S","Market Cap","Enterprise Value"],
                "Value":[
                    fmt_num(valuation["pe"]), fmt_num(valuation["forward_pe"]),
                    fmt_num(valuation["peg"]), fmt_num(valuation["pb"]),
                    fmt_num(valuation["ev_ebitda"]), fmt_num(valuation["ps"]),
                    fmt_num(valuation["market_cap"]), fmt_num(valuation["enterprise_value"])
                ],
            }),
            use_container_width=True,
            hide_index=True
        )
        st.markdown(f"**Valuation score: {valuation['score']:.0f}/100**")
        missing_val = [k for k,v in {
            "Trailing P/E": valuation["pe"],
            "Forward P/E": valuation["forward_pe"],
            "PEG": valuation["peg"],
            "P/B": valuation["pb"],
            "EV/EBITDA": valuation["ev_ebitda"],
            "P/S": valuation["ps"],
        }.items() if pd.isna(v)]
        if missing_val:
            st.caption("Unavailable valuation fields: " + ", ".join(missing_val) +
                       ". Forward metrics require provider estimates; other gaps usually indicate missing shares/price/statement inputs.")
        st.bar_chart(pd.Series(valuation["parts"], name="Valuation score", dtype="float64"))
        st.caption("Relative valuation is calculated from available price + financial-statement data. Missing forward metrics remain explicitly unavailable.")


    with tabs[5]:
        if hist.empty:
            st.warning(
                "Technical analysis is unavailable because the current public market-data "
                "providers returned no reliable OHLCV history for this security."
            )
            st.info(
                "V3.0 continues with fundamentals, valuation, risk flags and evidence analysis. "
                "Technical score is excluded from the overall score when OHLCV is unavailable."
            )
        else:
            latest=tech.iloc[-1]
            st.line_chart(tech[["Close","SMA20","SMA50","SMA100","SMA200"]].tail(300))
            a,b,c,d,e=st.columns(5)
            a.metric("RSI(14)",fmt_num(latest["RSI14"]))
            b.metric("ADX(14)",fmt_num(latest["ADX14"]))
            c.metric("ATR(14)",f"₹{fmt_num(latest['ATR14'])}")
            d.metric("MACD",fmt_num(latest["MACD"],2))
            e.metric("Volume ratio",fmt_num(latest["VolumeRatio"],2))
            st.bar_chart(pd.Series(technical_components,name="Points",dtype="float64"))
            atr=safe_num(latest["ATR14"])
            if not pd.isna(atr):
                stop=price-1.5*atr
                risk_per_share=max(price-stop,.01)
                risk_amount=capital*risk_pct/100
                qty=int(risk_amount/risk_per_share)
                target1=price+2*risk_per_share
                target2=price+3*risk_per_share
                a,b,c,d=st.columns(4)
                a.metric("Illustrative stop",f"₹{stop:,.2f}")
                b.metric("Risk budget",f"₹{risk_amount:,.0f}")
                c.metric("Risk-based qty",f"{qty:,}")
                d.metric("R:R targets",f"₹{target1:,.2f} / ₹{target2:,.2f}")
                st.caption("Illustrative only; liquidity, gaps and slippage are not included.")

    with tabs[6]:
        st.markdown("### Automated red flags")
        for x in flags: st.error("⚠ "+x)
        if not flags: st.success("No major automated red flags triggered.")
        st.markdown("### Positive catalysts / quality signals")
        for x in positives: st.success("✓ "+x)
        if not positives: st.write("None triggered.")
        st.markdown("### Exchange / data caveats")
        st.write("SME securities can have lower liquidity, larger spreads, market-maker/call-auction mechanics and shorter public operating histories. Position sizing should therefore use liquidity-aware limits rather than the mechanical ATR quantity alone.")

    with tabs[7]:
        st.caption("News is currently a supplementary yfinance feed; it is not used as the authoritative exchange-data source.")
        if data["news"]:
            for n in data["news"]:
                title=n.get("title") or "Untitled"; publisher=n.get("publisher") or ""; url=n.get("url")
                st.markdown(f"**{title}** — {publisher}")
                if url: st.markdown(f"[Open source]({url})")
        else: st.info("No recent news was returned by the available news feed.")


    with tabs[8]:
        st.markdown("### Data availability")
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Security identity", "VERIFIED")
        d2.metric("Market data", "AVAILABLE" if not hist.empty else "LIMITED")
        d3.metric("Fundamentals", "AVAILABLE" if not data["annual_income"].empty else "LIMITED")
        d4.metric("Management evidence", management["confidence"].split(" — ")[0])

        st.markdown("### Security identity")
        st.dataframe(pd.DataFrame([{
            "Exchange": sec.get("exchange"),
            "Segment": sec.get("segment"),
            "Symbol": sec.get("symbol"),
            "BSE code": sec.get("bse_code"),
            "ISIN": sec.get("isin"),
            "Universe source": sec.get("source"),
        }]), use_container_width=True, hide_index=True)

        st.markdown("### Market snapshot")
        a,b,c,d = st.columns(4)
        a.metric("Current price", fmt_num(valuation.get("price")))
        b.metric("Shares", fmt_num(valuation.get("shares")))
        c.metric("Market cap", fmt_num(valuation.get("market_cap")))
        d.metric("Enterprise value", fmt_num(valuation.get("enterprise_value")))

        st.markdown("### Market-data provenance")
        st.write({
            "Primary provider used": data.get("provider"),
            "Provider chain": data.get("provider_chain"),
            "Historical sessions": len(hist),
            "History period": data.get("history_period"),
            "Data quality": info.get("dataQuality", "LOW"),
        })

        st.markdown("### Public source links")
        for label, url in source_links(sec):
            st.markdown(f"- [{label}]({url})")

        st.markdown("### Evidence documents uploaded in this session")
        if uploaded_files:
            for doc in uploaded_files:
                st.write(f"• {doc['name']} (SHA-256 prefix: {doc['sha']})")
        else:
            st.info("No company documents uploaded. Management/guidance score is therefore an evidence-availability score, not a claim about management quality.")

        st.markdown("### Evidence hits")
        if management["findings"]:
            st.dataframe(
                pd.DataFrame(management["findings"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No document evidence was extracted yet.")

    with tabs[9]:
        st.markdown("### Annual balance sheet"); st.dataframe(data["annual_balance"],use_container_width=True)
        st.markdown("### Quarterly balance sheet"); st.dataframe(data["quarterly_balance"],use_container_width=True)
        st.markdown("### Annual cash flow"); st.dataframe(data["annual_cashflow"],use_container_width=True)
        st.markdown("### Quarterly cash flow"); st.dataframe(data["quarterly_cashflow"],use_container_width=True)

    st.divider()
    st.caption(
    "V3.2 is an analytical prototype, not investment advice. No broker account is required. "
    "Exchange identity is kept separate from market-data availability; public data can still be incomplete, "
    "especially for SME securities. Verify material decisions against exchange/company filings."
)

except Exception as exc:
    st.error(f"Could not analyse {stock}. Error: {type(exc).__name__}: {exc}")
    st.exception(exc)
