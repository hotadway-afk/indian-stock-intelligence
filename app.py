import io
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Indian Stock Intelligence V2.6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# V2.6: Exchange-first market-data architecture
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

    # Critical V2.6 change: direct BSE resolution is attempted even when the
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
    """Legacy fallback ticker only. Exchange data is preferred in V2.6."""
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
    """V2.6 exchange-first data layer.

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

    info = {}
    annual_income = quarterly_income = annual_balance = quarterly_balance = annual_cashflow = quarterly_cashflow = pd.DataFrame()
    news = []

    # yfinance is used for structured fundamentals only when available; exchange identity stays authoritative.
    try:
        info = dict(ticker.info or {})
    except Exception:
        info = {}
    try:
        annual_income = normalize_statement(ticker.get_income_stmt(freq="yearly"))
        quarterly_income = normalize_statement(ticker.get_income_stmt(freq="quarterly"))
        annual_balance = normalize_statement(ticker.get_balance_sheet(freq="yearly"))
        quarterly_balance = normalize_statement(ticker.get_balance_sheet(freq="quarterly"))
        annual_cashflow = normalize_statement(ticker.get_cash_flow(freq="yearly"))
        quarterly_cashflow = normalize_statement(ticker.get_cash_flow(freq="quarterly"))
    except Exception:
        pass
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
# Fundamental / valuation / quality engines
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
    roe = safe_num(info.get("returnOnEquity")); roa = safe_num(info.get("returnOnAssets")); margin = safe_num(info.get("profitMargins"))
    rev_growth = safe_num(info.get("revenueGrowth")); earnings_growth = safe_num(info.get("earningsGrowth")); de = safe_num(info.get("debtToEquity"))
    fcf = np.nan
    if not pd.isna(operating_cf) and not pd.isna(capex):
        fcf = operating_cf + capex if capex < 0 else operating_cf - capex
    components = {
        "ROE": score_range(roe, 0.08, 0.25), "ROA": score_range(roa, 0.03, 0.15),
        "Revenue growth": score_range(rev_growth, 0.00, 0.25), "Earnings growth": score_range(earnings_growth, 0.00, 0.30),
        "Profit margin": score_range(margin, 0.05, 0.25), "Debt discipline": score_range(de, 20, 150, inverse=True),
    }
    return {"score": float(np.mean(list(components.values()))), "components": components, "revenue": revenue, "net_income": net_income,
            "ebitda": ebitda, "operating_cf": operating_cf, "capex": capex, "fcf": fcf, "debt": debt, "equity": equity, "cash": cash,
            "roe": roe, "roa": roa, "margin": margin, "rev_growth": rev_growth, "earnings_growth": earnings_growth, "de": de}


def valuation_engine(info, fundamental):
    pe = safe_num(info.get("trailingPE")); fpe = safe_num(info.get("forwardPE")); peg = safe_num(info.get("pegRatio")); pb = safe_num(info.get("priceToBook")); ev_ebitda = safe_num(info.get("enterpriseToEbitda")); ps = safe_num(info.get("priceToSalesTrailing12Months"))
    parts = {"P/E": 50 if pd.isna(pe) else score_range(pe,10,60,True), "Forward P/E": 50 if pd.isna(fpe) else score_range(fpe,8,50,True),
             "PEG": 50 if pd.isna(peg) else score_range(peg,.5,3,True), "P/B": 50 if pd.isna(pb) else score_range(pb,1,10,True),
             "EV/EBITDA": 50 if pd.isna(ev_ebitda) else score_range(ev_ebitda,5,35,True)}
    return {"score": float(np.mean(list(parts.values()))), "parts": parts, "pe": pe, "forward_pe": fpe, "peg": peg, "pb": pb, "ev_ebitda": ev_ebitda, "ps": ps}


def quality_engine(info, fundamental):
    flags=[]; positives=[]
    if not pd.isna(fundamental["roe"]) and fundamental["roe"] >= .15: positives.append(f"ROE is {fundamental['roe']*100:.1f}%, above 15%.")
    if not pd.isna(fundamental["rev_growth"]):
        g=fundamental["rev_growth"]
        if g >= .15: positives.append(f"Revenue grew {g*100:.1f}% YoY — strong growth.")
        elif g >= .05: positives.append(f"Revenue grew {g*100:.1f}% YoY — moderate growth.")
        elif g >= 0: flags.append(f"Revenue growth is only {g*100:.1f}% YoY — weak growth.")
        else: flags.append(f"Revenue declined {abs(g)*100:.1f}% YoY.")
    if not pd.isna(fundamental["fcf"]) and fundamental["fcf"] > 0: positives.append("Latest annual cash flow indicates positive free cash flow.")
    if not pd.isna(fundamental["de"]) and fundamental["de"] > 150: flags.append("Debt/equity is elevated.")
    if not pd.isna(fundamental["earnings_growth"]) and fundamental["earnings_growth"] < 0: flags.append(f"Earnings growth is negative ({fundamental['earnings_growth']*100:.1f}%).")
    if not pd.isna(fundamental["fcf"]) and not pd.isna(fundamental["net_income"]) and fundamental["fcf"] < 0 and fundamental["net_income"] > 0:
        flags.append("Positive accounting profit with negative latest annual free cash flow; investigate earnings quality/capex.")
    promoter=safe_num(info.get("heldPercentInsiders"))
    if not pd.isna(promoter) and promoter < .20: flags.append("Insider/promoter ownership is below 20% in the available snapshot.")
    return max(0, min(100, 100-12*len(flags)+6*len(positives))), positives, flags


# -----------------------------
# Management/evidence placeholder — explicit, never fabricated
# -----------------------------
def management_engine(sec, info):
    # V2.3 separates evidence availability from management quality.
    # Company-specific guidance still requires filings/IR documents.
    evidence=[]
    if sec.get("exchange"): evidence.append(f"Exchange universe verified: {sec.get('exchange')} / {sec.get('segment')}.")
    if info.get("exchangeISIN"): evidence.append("ISIN verified from exchange universe.")
    provider=info.get("dataProvider")
    evidence.append(f"Market-data provider: {provider}.")
    return {
        "score": np.nan,
        "confidence": "LOW — filing-level management evidence not ingested",
        "evidence": evidence,
        "guidance": "Not verified",
        "note": "V2.6 does not manufacture FY27/FY28 guidance. Add NSE/BSE filings, investor presentations and earnings-call transcripts for a company-specific management score."
    }


# -----------------------------
# UI
# -----------------------------
st.title("📊 Indian Stock Intelligence — V2.6")
st.caption("Exchange-first NSE + BSE + NSE Emerge + BSE SME Fundamental + Evidence + Valuation + Technical + Risk engine")

universe = load_universe()

with st.sidebar:
    st.header("Analysis Inputs")
    exchange_choice = st.selectbox("Exchange / segment", ["All Exchanges", "NSE", "BSE", "NSE Emerge / SME", "BSE SME"])
    stock = st.text_input("Symbol / BSE code / ISIN / company", "RELIANCE").strip().upper()
    capital = st.number_input("Portfolio capital (₹)", min_value=10000, value=500000, step=10000)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    objective = st.selectbox("Primary objective", ["Long Term", "Swing Trading", "Intraday"])
    run = st.button("🚀 RUN V2.6 ANALYSIS", type="primary", use_container_width=True)

if not run:
    st.info("Enter a symbol, BSE code, ISIN or company name and click RUN V2.6 ANALYSIS.")
    a,b,c,d = st.columns(4)
    nse_count = len(universe[universe.exchange == "NSE"]) if not universe.empty else 0
    bse_count = len(universe[universe.exchange == "BSE"]) if not universe.empty else 0
    nse_sme = len(universe[universe.segment == "NSE Emerge / SME"]) if not universe.empty else 0
    bse_sme = len(universe[universe.segment == "BSE SME"]) if not universe.empty else 0
    a.metric("NSE universe", f"{nse_count:,}")
    b.metric("BSE universe", f"{bse_count:,}")
    c.metric("NSE Emerge / SME", f"{nse_sme:,}")
    d.metric("BSE SME", f"{bse_sme:,}")
    st.markdown("### V2.6 coverage")
    st.markdown("- NSE Main Board")
    st.markdown("- NSE Emerge / SME")
    st.markdown("- BSE Main Board — all active equity groups")
    st.markdown("- BSE SME — M / MT / MS / TS groups")
    st.markdown("- Symbol, BSE scrip code, ISIN and company-name lookup")
    st.markdown("- Exchange data first (NSE/BSE); yfinance only as legacy fallback")
    st.caption("V2.6 separates exchange identity from provider coverage. BSE direct identity resolution is attempted even if the bulk BSE master endpoint is unavailable.")
    st.stop()

try:
    sec, matches = resolve_security(stock, exchange_choice, universe)
    if sec is None:
        st.error(f"Could not resolve '{stock}' in the selected exchange/segment universe.")
        st.info("V2.6 checks the exchange universe first and then attempts direct BSE identity resolution. If this still fails, the exchange provider did not return a usable identity for the input.")
        if not matches.empty:
            st.dataframe(matches[["exchange","segment","symbol","company","isin","bse_code","source"]] if "source" in matches.columns else matches[["exchange","segment","symbol","company","isin","bse_code"]], use_container_width=True, hide_index=True)
        st.stop()

    if len(matches) > 1 and stock not in [str(sec.get("symbol","")), str(sec.get("bse_code",""))]:
        st.info("Multiple matches found. The first exact/best match is being analysed.")
        st.dataframe(matches[["exchange","segment","symbol","company","isin","bse_code","source"]] if "source" in matches.columns else matches[["exchange","segment","symbol","company","isin","bse_code"]], use_container_width=True, hide_index=True)

    with st.spinner(f"Loading {sec.get('company') or stock} data..."):
        data = load_stock(sec)

    hist = data["hist"]; info = data["info"]
    if hist.empty:
        st.error("The security was resolved from the exchange universe, but no usable historical market-data series was returned by NSE/BSE or the legacy fallback.")
        st.warning("This is a market-data coverage limitation — NOT a statement that the company is unlisted.")
        st.dataframe(pd.DataFrame([{"Exchange":sec.get("exchange"),"Segment":sec.get("segment"),"Symbol":sec.get("symbol"),"BSE code":sec.get("bse_code"),"ISIN":sec.get("isin"),"Legacy fallback ticker":info.get("providerTicker"),"Universe source":sec.get("source")}]),use_container_width=True,hide_index=True)
        st.stop()

    tech, technical_score, technical_components = technical_engine(hist)
    fundamental = fundamental_engine(info, data["annual_income"], data["annual_balance"], data["annual_cashflow"])
    valuation = valuation_engine(info, fundamental)
    quality_score, positives, flags = quality_engine(info, fundamental)
    management = management_engine(sec, info)
    management_display = 50.0 if pd.isna(management["score"]) else management["score"]

    overall = 0.35*fundamental["score"] + 0.20*management_display + 0.15*valuation["score"] + 0.20*technical_score + 0.10*quality_score
    if objective == "Long Term":
        objective_score = 0.45*fundamental["score"] + 0.25*management_display + 0.20*valuation["score"] + 0.10*quality_score
    elif objective == "Swing Trading":
        objective_score = 0.55*technical_score + 0.20*valuation["score"] + 0.15*quality_score + 0.10*fundamental["score"]
    else:
        objective_score = 0.70*technical_score + 0.20*quality_score + 0.10*valuation["score"]

    if objective_score >= 75: verdict="🟢 STRONG SETUP"
    elif objective_score >= 60: verdict="🟡 SELECTIVE / WATCH"
    elif objective_score >= 45: verdict="🟠 WAIT / NEUTRAL"
    else: verdict="🔴 AVOID / HIGH RISK"

    company = info.get("longName") or sec.get("company") or stock
    price = safe_num(tech.iloc[-1]["Close"])
    high52 = safe_num(info.get("fiftyTwoWeekHigh")); low52 = safe_num(info.get("fiftyTwoWeekLow"))
    if pd.isna(high52) and not pd.isna(price): high52 = safe_num(tech["High"].tail(252).max())
    if pd.isna(low52) and not pd.isna(price): low52 = safe_num(tech["Low"].tail(252).min())

    st.subheader(f"{company} ({sec.get('symbol') or stock})")
    st.caption(f"{sec.get('exchange')} • {sec.get('segment')} • BSE code: {sec.get('bse_code') or '—'} • ISIN: {sec.get('isin') or '—'} • Universe: {sec.get('source') or '—'} • Data: {data['provider']} ({data['history_period']})")

    with st.expander("🔎 Security identity / exchange verification", expanded=True):
        id1,id2,id3,id4=st.columns(4)
        id1.metric("Exchange",str(sec.get("exchange") or "N/A")); id2.metric("Segment",str(sec.get("segment") or "N/A")); id3.metric("BSE code",str(sec.get("bse_code") or "N/A")); id4.metric("ISIN",str(sec.get("isin") or "N/A"))
        if sec.get("segment_note"): st.warning(sec.get("segment_note"))
        st.caption("Exchange identity is authoritative. V2.6 attempts NSE/BSE exchange data first; yfinance remains a legacy fallback only. Missing provider history is never treated as proof of an unlisted security.")
        st.markdown("#### Market-data provenance")
        st.write({
            "Primary provider used": data.get("provider"),
            "Provider chain attempted": data.get("provider_chain"),
            "History available": f"{len(data.get('hist', pd.DataFrame())):,} sessions",
            "History period": data.get("history_period"),
            "Data quality": info.get("dataQuality", "LOW"),
        })

    cols=st.columns(6)
    cols[0].metric("V2.6 Score", f"{overall:.0f}/100")
    cols[1].metric("Fundamental", f"{fundamental['score']:.0f}")
    cols[2].metric("Management", "N/A")
    cols[3].metric("Valuation", f"{valuation['score']:.0f}")
    cols[4].metric("Technical", f"{technical_score:.0f}")
    cols[5].metric("Quality/Risk", f"{quality_score:.0f}")

    if "STRONG" in verdict: st.success(verdict)
    elif "WATCH" in verdict: st.warning(verdict)
    elif "WAIT" in verdict: st.info(verdict)
    else: st.error(verdict)
    st.caption(f"Primary objective: {objective} | Objective-specific score: {objective_score:.0f}/100")

    tabs=st.tabs(["🎯 Executive Decision","🏢 Fundamentals","📊 Quarterly","🧠 Management / Evidence","💰 Valuation","📈 Technicals","🚨 Risks & Catalysts","📰 Recent News","📋 Raw Financials"])

    with tabs[0]:
        a,b,c,d=st.columns(4)
        a.metric("Price", f"₹{price:,.2f}")
        b.metric("52W High", f"₹{high52:,.2f}" if not pd.isna(high52) else "N/A")
        c.metric("52W Low", f"₹{low52:,.2f}" if not pd.isna(low52) else "N/A")
        mcap=info.get("marketCap")
        d.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
        st.markdown("### Coverage & decision framework")
        st.write(f"Exchange identity is resolved independently from market data. **OHLCV source:** {data.get('provider')} | **History:** {len(hist):,} sessions | **Quality:** {info.get('dataQuality', 'LOW')}.")
        st.markdown("### Key positives")
        for p in positives: st.success("✓ "+p)
        if not positives: st.write("No strong positive flags identified by the current structured-data rules.")
        st.markdown("### Key concerns")
        for f in flags: st.error("⚠ "+f)
        if not flags: st.success("No major automated red flags triggered.")

    with tabs[1]:
        st.markdown("### Fundamental scorecard")
        st.dataframe(pd.DataFrame({"Metric":["ROE","ROA","Revenue growth","Earnings growth","Profit margin","Debt/Equity","Latest annual FCF"],"Value":[fmt_pct(fundamental['roe']),fmt_pct(fundamental['roa']),fmt_pct(fundamental['rev_growth']),fmt_pct(fundamental['earnings_growth']),fmt_pct(fundamental['margin']),fmt_num(fundamental['de']),f"₹{fundamental['fcf']/1e7:,.1f} Cr" if not pd.isna(fundamental['fcf']) else "N/A"]}),use_container_width=True,hide_index=True)
        st.bar_chart(pd.Series(fundamental["components"],dtype="float64"))
        st.markdown("### Annual income statement")
        st.dataframe(data["annual_income"],use_container_width=True)

    with tabs[2]:
        st.markdown("### Quarterly income statement")
        st.dataframe(data["quarterly_income"],use_container_width=True)
        st.markdown("### Quarterly balance sheet")
        st.dataframe(data["quarterly_balance"],use_container_width=True)
        st.markdown("### Quarterly cash flow")
        st.dataframe(data["quarterly_cashflow"],use_container_width=True)

    with tabs[3]:
        st.markdown("### Management / evidence discipline")
        st.info(management["note"])
        a,b=st.columns(2)
        a.metric("Management score", "N/A")
        b.metric("Evidence confidence", "LOW")
        st.markdown("### Verified structured evidence")
        for e in management["evidence"]: st.write("• "+e)
        st.markdown("### Guidance tracker")
        st.write("FY27 revenue guidance: Not verified")
        st.write("FY27 EBITDA / margin guidance: Not verified")
        st.write("FY28 revenue guidance: Not verified")
        st.write("FY28 EBITDA / margin guidance: Not verified")
        st.write("Order book / capex / capacity guidance: Not verified")
        st.markdown("### What V2.6 should ingest")
        st.write("NSE/BSE announcements, annual reports, investor presentations, earnings-call transcripts and company IR documents, with date/source citations and guidance-vs-actual tracking.")

    with tabs[4]:
        st.markdown("### Relative valuation")
        st.dataframe(pd.DataFrame({"Metric":["Trailing P/E","Forward P/E","PEG","P/B","EV/EBITDA","P/S"],"Value":[fmt_num(valuation['pe']),fmt_num(valuation['forward_pe']),fmt_num(valuation['peg']),fmt_num(valuation['pb']),fmt_num(valuation['ev_ebitda']),fmt_num(valuation['ps'])]}),use_container_width=True,hide_index=True)
        st.bar_chart(pd.Series(valuation["parts"],name="Valuation score",dtype="float64"))
        st.caption("No fabricated DCF is shown. A DCF requires explicit normalized earnings/cash-flow assumptions and scenarios.")

    with tabs[5]:
        latest=tech.iloc[-1]
        st.line_chart(tech[["Close","SMA20","SMA50","SMA100","SMA200"]].tail(300))
        a,b,c,d,e=st.columns(5)
        a.metric("RSI(14)",fmt_num(latest["RSI14"])); b.metric("ADX(14)",fmt_num(latest["ADX14"])); c.metric("ATR(14)",f"₹{fmt_num(latest['ATR14'])}"); d.metric("MACD",fmt_num(latest["MACD"],2)); e.metric("Volume ratio",fmt_num(latest["VolumeRatio"],2))
        st.bar_chart(pd.Series(technical_components,name="Points",dtype="float64"))
        atr=safe_num(latest["ATR14"])
        if not pd.isna(atr):
            stop=price-1.5*atr; risk_per_share=max(price-stop,.01); risk_amount=capital*risk_pct/100; qty=int(risk_amount/risk_per_share); target1=price+2*risk_per_share; target2=price+3*risk_per_share
            a,b,c,d=st.columns(4); a.metric("Illustrative stop",f"₹{stop:,.2f}"); b.metric("Risk budget",f"₹{risk_amount:,.0f}"); c.metric("Risk-based qty",f"{qty:,}"); d.metric("R:R targets",f"₹{target1:,.2f} / ₹{target2:,.2f}")
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
        st.markdown("### Annual balance sheet"); st.dataframe(data["annual_balance"],use_container_width=True)
        st.markdown("### Quarterly balance sheet"); st.dataframe(data["quarterly_balance"],use_container_width=True)
        st.markdown("### Annual cash flow"); st.dataframe(data["annual_cashflow"],use_container_width=True)
        st.markdown("### Quarterly cash flow"); st.dataframe(data["quarterly_cashflow"],use_container_width=True)

    st.divider()
    st.caption("V2.6 is an analytical prototype, not investment advice. Exchange identity and security masters are sourced from official exchange endpoints where available; free market-data providers can be incomplete, especially for SME securities. Verify material decisions against exchange/company filings.")

except Exception as exc:
    st.error(f"Could not analyse {stock}. Error: {type(exc).__name__}: {exc}")
    st.exception(exc)
