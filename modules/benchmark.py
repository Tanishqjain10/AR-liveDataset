import pandas as pd
import yfinance as yf

from .database import bulk_upsert, read_sql


def fetch_nifty_history() -> pd.DataFrame:
    try:
        hist = yf.download(
            "^NSEI",
            period="max",
            progress=False,
            auto_adjust=False
        )

        if hist is None or hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()

        hist.columns = [
            str(c).lower().replace(" ", "_")
            for c in hist.columns
        ]

        hist["benchmark_name"] = "Nifty 50"
        hist["benchmark_symbol"] = "^NSEI"
        hist["source"] = "Yahoo Finance"

        hist["date"] = pd.to_datetime(hist["date"]).dt.date.astype(str)

        return hist

    except Exception as e:
        print("NIFTY ERROR:", e)
        return pd.DataFrame()


def store_nifty_history(df: pd.DataFrame):
    if df.empty:
        return
    rows = []
    for _, r in df.iterrows():
        def safe(k):
            v = r.get(k)
            return float(v) if v is not None and str(v) not in {"", "nan", "None"} else None
        rows.append({
            "benchmark_name": r.get("benchmark_name", "Nifty 50"),
            "benchmark_symbol": r.get("benchmark_symbol", "^NSEI"),
            "source": r.get("source", "Yahoo Finance"),
            "date": str(r.get("date", "")),
            "open": safe("open"),
            "high": safe("high"),
            "low": safe("low"),
            "close": safe("close"),
            "adj_close": safe("adj_close"),
            "volume": safe("volume"),
        })
    if rows:
        bulk_upsert("benchmark", rows, conflict_cols=["benchmark_name", "date"])


def get_benchmark_df() -> pd.DataFrame:
    rows = read_sql("SELECT date, close FROM benchmark WHERE benchmark_symbol='^NSEI' ORDER BY date")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def compute_benchmark_returns(df: pd.DataFrame) -> dict:
    if df.empty or "close" not in df.columns:
        return {}
    series = df["close"].dropna().sort_index()
    if series.empty:
        return {}

    def ptp(days):
        if len(series) > days:
            s, e = series.iloc[-days], series.iloc[-1]
            return round((e / s - 1) * 100, 2) if s > 0 else None
        return None

    def cagr(years):
        days = int(years * 252)
        if len(series) > days:
            s = series.iloc[-days]
            e = series.iloc[-1]
            start_date = series.index[-days]
            actual_years = (series.index[-1] - start_date).days / 365.25
            if s > 0 and actual_years > 0:
                return round(((e / s) ** (1 / actual_years) - 1) * 100, 2)
        return None

    return {
        "1M": ptp(21), "3M": ptp(63), "6M": ptp(126),
        "1Y": ptp(252), "3Y": cagr(3), "5Y": cagr(5),
    }
