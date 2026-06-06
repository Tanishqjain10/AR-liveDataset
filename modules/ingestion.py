import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .cleaning import clean_num, clean_text, parse_date_safe
from .database import bulk_upsert, execute, init_db, read_sql, to_json, upsert

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SchemeRef:
    scheme_code: str
    scheme_name: str
    url: str
    category: str | None = None
    section: str | None = None
    raw_scheme_slug: str | None = None


def discover_schemes(urls: list[str]) -> list[SchemeRef]:
    discovered = {}
    for u in urls:
        p = urlparse(u)
        parts = [x for x in p.path.split("/") if x]
        if "funds" in parts:
            idx = parts.index("funds")
            scheme_code = parts[idx + 1] if idx + 1 < len(parts) else None
            scheme_slug = parts[idx + 2] if idx + 2 < len(parts) else ""
            if scheme_code:
                base = f"{p.scheme}://{p.netloc}/funds/{scheme_code}/{scheme_slug}/"
                key = scheme_code
                section = p.fragment or "overview"
                if key not in discovered:
                    discovered[key] = SchemeRef(
                        scheme_code=scheme_code,
                        scheme_name=_infer_name_from_slug(scheme_slug),
                        url=base,
                        section=section,
                        raw_scheme_slug=scheme_slug,
                    )
    return list(discovered.values())


def _infer_name_from_slug(slug: str) -> str:
    slug = (slug or "").replace("-direct-plan", " Direct").replace("-regular-plan", " Regular")
    words = [w.capitalize() for w in slug.split("-") if w]
    return " ".join(words).strip() or slug


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_json(url: str, timeout: int = 20):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 MFAnalytics/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_mfapi_latest(scheme_code: str):
    for base in ("https://api.mfapi.in/mf", "https://www.mfapi.in/mf"):
        try:
            return fetch_json(f"{base}/{scheme_code}/latest")
        except Exception:
            continue
    return None


def fetch_mfapi_history(scheme_code: str):
    for base in ("https://api.mfapi.in/mf", "https://www.mfapi.in/mf"):
        try:
            return fetch_json(f"{base}/{scheme_code}")
        except Exception:
            continue
    return None


# ── NAV calculations ──────────────────────────────────────────────────────────

def _build_nav_series(history_data: list) -> pd.Series:
    """Convert MFAPI history list to a date-indexed NAV Series (ascending)."""
    if not history_data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(history_data)
    if "nav" not in df.columns or "date" not in df.columns:
        return pd.Series(dtype=float)
    df["parsed_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["parsed_date"])
    df["nav_val"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav_val"])
    series = df.set_index("parsed_date")["nav_val"].sort_index()
    return series[~series.index.duplicated(keep="last")]


def _ptp_return(series: pd.Series, days: int) -> float | None:
    if series.empty or len(series) < 2:
        return None
    end = series.iloc[-1]
    start = series.iloc[-days] if len(series) > days else series.iloc[0]
    if start == 0:
        return None
    return round((end / start - 1) * 100, 2)


def _cagr(series: pd.Series, years: float) -> float | None:
    days = int(years * TRADING_DAYS)
    if series.empty or len(series) < 2:
        return None
    end = series.iloc[-1]
    start = series.iloc[-days] if len(series) > days else series.iloc[0]
    start_date = series.index[-days] if len(series) > days else series.index[0]
    actual_years = (series.index[-1] - start_date).days / 365.25
    if start == 0 or actual_years <= 0:
        return None
    return round(((end / start) ** (1 / actual_years) - 1) * 100, 2)


def _since_inception_cagr(series: pd.Series) -> float | None:
    if series.empty or len(series) < 2:
        return None
    years = (series.index[-1] - series.index[0]).days / 365.25
    if series.iloc[0] == 0 or years <= 0:
        return None
    return round(((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100, 2)


def _compute_risk(fund_series: pd.Series, bench_series: pd.Series) -> dict:
    if fund_series.empty:
        return {}
    fr = fund_series.pct_change().dropna()
    br = bench_series.pct_change().dropna() if not bench_series.empty else pd.Series(dtype=float)

    std = float(fr.std() * np.sqrt(TRADING_DAYS) * 100) if len(fr) > 10 else None
    vol = std
    downside = float(fr[fr < 0].std() * np.sqrt(TRADING_DAYS) * 100) if len(fr[fr < 0]) > 5 else None

    beta, alpha, sharpe, sortino = None, None, None, None
    if not br.empty:
        aligned = pd.concat([fr, br], axis=1, join="inner").dropna()
        if len(aligned) > 20:
            f, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
            var_b = float(np.var(b))
            if var_b > 0:
                beta = float(np.cov(f, b)[0, 1] / var_b)
                daily_rf = RISK_FREE_RATE / TRADING_DAYS
                ann_fund = (f.mean() - daily_rf) * TRADING_DAYS * 100
                ann_bench = (b.mean() - daily_rf) * TRADING_DAYS * 100
                alpha = round(ann_fund - beta * ann_bench, 4)
                beta = round(beta, 4)

    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    excess = fr - daily_rf
    if fr.std() > 0 and len(fr) > 10:
        sharpe = round(float(excess.mean() / fr.std() * np.sqrt(TRADING_DAYS)), 4)
    if len(fr[fr < 0]) > 5 and fr[fr < 0].std() > 0:
        sortino = round(float(excess.mean() / fr[fr < 0].std() * np.sqrt(TRADING_DAYS)), 4)

    return {
        "standard_deviation": round(std, 4) if std else None,
        "beta": beta,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "alpha": alpha,
        "volatility": round(vol, 4) if vol else None,
        "downside_volatility": round(downside, 4) if downside else None,
    }


# ── Benchmark fetch ───────────────────────────────────────────────────────────

def fetch_benchmark_series() -> pd.Series:
    rows = read_sql("SELECT date, close FROM benchmark ORDER BY date")
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


# ── AMFI monthly flow ingest ──────────────────────────────────────────────────

def _ingest_flows_from_nav(scheme_code: str, scheme_name: str, nav_series: pd.Series):
    """Derive monthly relative flow proxy from NAV changes as best-effort approximation."""
    if nav_series.empty:
        return
    monthly = nav_series.resample("ME").last().dropna()
    rows = []
    for i in range(1, len(monthly)):
        prev = monthly.iloc[i - 1]
        curr = monthly.iloc[i]
        period = monthly.index[i].strftime("%Y-%m")
        ret = (curr - prev) / prev if prev > 0 else 0
        net = round(ret * 100, 4)
        rows.append({
            "scheme_code": scheme_code,
            "scheme_name": scheme_name,
            "category": None,
            "period": period,
            "flow_type": "net_nav_proxy",
            "amount": net,
            "source": "NAV-derived",
            "source_url": "https://api.mfapi.in/mf",
            "last_updated": utc_now(),
            "source_status": "Computed",
        })
    if rows:
        bulk_upsert("flows", rows, conflict_cols=["scheme_code", "period", "flow_type", "source"])


# ── Portfolio changes ─────────────────────────────────────────────────────────

def _compute_and_store_portfolio_changes(scheme_code: str, scheme_name: str):
    rows = read_sql(
        "SELECT as_of_date FROM holdings WHERE scheme_code=? GROUP BY as_of_date ORDER BY as_of_date DESC LIMIT 2",
        [scheme_code]
    )
    dates = [r["as_of_date"] for r in rows]
    if len(dates) < 2:
        return
    curr_date, prev_date = dates[0], dates[1]

    curr_rows = read_sql("SELECT * FROM holdings WHERE scheme_code=? AND as_of_date=?", [scheme_code, curr_date])
    prev_rows = read_sql("SELECT * FROM holdings WHERE scheme_code=? AND as_of_date=?", [scheme_code, prev_date])

    curr_map = {(r["isin"] or r["holding_name"]): dict(r) for r in curr_rows}
    prev_map = {(r["isin"] or r["holding_name"]): dict(r) for r in prev_rows}
    all_keys = set(curr_map) | set(prev_map)

    for key in all_keys:
        c = curr_map.get(key)
        p = prev_map.get(key)
        cw = clean_num(c["weight"]) if c else 0.0
        pw = clean_num(p["weight"]) if p else 0.0
        if cw is None:
            cw = 0.0
        if pw is None:
            pw = 0.0

        if pw == 0 and cw > 0:
            ctype = "NEW_ADDITION"
        elif cw == 0 and pw > 0:
            ctype = "COMPLETE_EXIT"
        elif cw > pw:
            ctype = "INCREASED"
        elif cw < pw:
            ctype = "REDUCED"
        else:
            continue

        execute(
            """INSERT OR IGNORE INTO portfolio_changes
               (scheme_code, scheme_name, current_as_of_date, previous_as_of_date,
                holding_name, isin, change_type, weight_change, current_weight, previous_weight)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                scheme_code, scheme_name, curr_date, prev_date,
                (c or p)["holding_name"], (c or p).get("isin"),
                ctype, round(cw - pw, 4), cw, pw,
            ],
        )


# ── Main ingest orchestrator ──────────────────────────────────────────────────

def ingest_schemes(urls: list[str]):
    init_db()
    schemes = discover_schemes(urls)

    # Load benchmark once
    bench_series = fetch_benchmark_series()

    for s in schemes:
        try:
            nav_data = fetch_mfapi_latest(s.scheme_code)
            hist_data = fetch_mfapi_history(s.scheme_code)

            latest_nav, latest_nav_date = None, None
            if nav_data and isinstance(nav_data, dict):
                d = nav_data.get("data", nav_data)
                if isinstance(d, list) and d:
                    d = d[0]
                latest_nav = clean_num(d.get("nav") if isinstance(d, dict) else None)
                latest_nav_date = parse_date_safe(d.get("date") if isinstance(d, dict) else None)

            history_list = []
            meta_info = {}
            if hist_data and isinstance(hist_data, dict):
                history_list = hist_data.get("data", []) if isinstance(hist_data.get("data"), list) else []
                meta_info = hist_data.get("meta", {}) or {}

            nav_series = _build_nav_series(history_list)

            # Store NAV history
            if not nav_series.empty:
                nav_rows = [
                    {"scheme_code": s.scheme_code, "nav_date": idx.strftime("%Y-%m-%d"), "nav": float(val)}
                    for idx, val in nav_series.items()
                ]
                bulk_upsert("nav_history", nav_rows, conflict_cols=["scheme_code", "nav_date"])

            # Compute returns
            ret_1m = _ptp_return(nav_series, 21)
            ret_3m = _ptp_return(nav_series, 63)
            ret_6m = _ptp_return(nav_series, 126)
            ret_1y = _ptp_return(nav_series, 252)
            cagr_3y = _cagr(nav_series, 3)
            cagr_5y = _cagr(nav_series, 5)
            since_inc = _since_inception_cagr(nav_series)

            # Compute risk
            risk = _compute_risk(nav_series, bench_series)

            # Launch date from meta
            launch_raw = meta_info.get("scheme_start_date", "")
            launch_date = parse_date_safe(launch_raw) or parse_date_safe(
                nav_series.index[0].strftime("%Y-%m-%d") if not nav_series.empty else None
            )

            fund_row = {
                "scheme_code": s.scheme_code,
                "scheme_name": s.scheme_name,
                "category": clean_text(meta_info.get("scheme_category", s.category)),
                "plan_type": "Direct" if "direct" in (s.raw_scheme_slug or "").lower() else "Regular",
                "amc": clean_text(meta_info.get("fund_house", "")),
                "launch_date": launch_date,
                "fund_manager": None,
                "aum": None,
                "expense_ratio": None,
                "nav": latest_nav,
                "nav_date": latest_nav_date,
                "since_inception_return": since_inc,
                "one_month_return": ret_1m,
                "three_month_return": ret_3m,
                "six_month_return": ret_6m,
                "one_year_return": ret_1y,
                "three_year_cagr": cagr_3y,
                "five_year_cagr": cagr_5y,
                "std_dev": risk.get("standard_deviation"),
                "beta": risk.get("beta"),
                "sharpe_ratio": risk.get("sharpe_ratio"),
                "sortino_ratio": risk.get("sortino_ratio"),
                "alpha": risk.get("alpha"),
                "category_rank": None,
                "holdings_count": None,
                "benchmark_name": "Nifty 50",
                "benchmark_symbol": "^NSEI",
                "source_url": s.url,
                "source_primary": "MFAPI (Live NAV + History)",
                "source_secondary": "Yahoo Finance (Benchmark)",
                "last_updated": utc_now(),
                "source_status": "Live" if latest_nav is not None else "Unavailable",
                "raw_json": to_json({
                    "mfapi_latest": nav_data,
                    "history_count": len(history_list),
                    "meta": meta_info,
                }),
            }
            upsert("funds", fund_row, conflict_cols=["scheme_code"])

            # Risk metrics row
            today = date.today().isoformat()
            risk_row = {
                "scheme_code": s.scheme_code,
                "scheme_name": s.scheme_name,
                "as_of_date": today,
                "standard_deviation": risk.get("standard_deviation"),
                "beta": risk.get("beta"),
                "sharpe_ratio": risk.get("sharpe_ratio"),
                "sortino_ratio": risk.get("sortino_ratio"),
                "alpha": risk.get("alpha"),
                "volatility": risk.get("volatility"),
                "downside_volatility": risk.get("downside_volatility"),
                "tracking_error": None,
                "information_ratio": None,
                "source": "MFAPI + Yahoo Finance",
                "source_url": s.url,
                "source_status": "Computed",
            }
            upsert("risk_metrics", risk_row, conflict_cols=["scheme_code", "as_of_date"])

            # Flows from NAV proxy
            _ingest_flows_from_nav(s.scheme_code, s.scheme_name, nav_series)

            # Portfolio changes
            _compute_and_store_portfolio_changes(s.scheme_code, s.scheme_name)

            # Data quality log
            status = "OK" if latest_nav is not None else "MISSING_NAV"
            upsert(
                "data_quality",
                {
                    "entity_type": "fund",
                    "entity_key": s.scheme_code,
                    "metric_name": "nav",
                    "source": "MFAPI",
                    "last_updated": utc_now(),
                    "status": status,
                    "notes": f"history_rows={len(history_list)}",
                },
                conflict_cols=["entity_type", "entity_key", "metric_name"],
            )

            time.sleep(0.3)

        except Exception as e:
            upsert(
                "data_quality",
                {
                    "entity_type": "fund",
                    "entity_key": s.scheme_code,
                    "metric_name": "ingestion",
                    "source": "MFAPI",
                    "last_updated": utc_now(),
                    "status": "ERROR",
                    "notes": str(e)[:500],
                },
                conflict_cols=["entity_type", "entity_key", "metric_name"],
            )

    # Assign category ranks by std_dev
    _assign_category_ranks()
    return schemes


def _assign_category_ranks():
    rows = read_sql("SELECT scheme_code, std_dev FROM funds WHERE std_dev IS NOT NULL ORDER BY std_dev ASC")
    for rank, row in enumerate(rows, 1):
        execute("UPDATE funds SET category_rank=? WHERE scheme_code=?", [rank, row["scheme_code"]])
