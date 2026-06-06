# 📊 Mutual Fund Portfolio Analytics Platform

A production-grade, fully live mutual fund analytics dashboard built with Python and Streamlit.

---

## 🗂 Folder Structure

```
mutual-fund-portfolio-analytics/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt
├── runtime.txt             # Python 3.11.9
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep            # portfolio.db is created at runtime
└── modules/
    ├── __init__.py
    ├── database.py         # SQLite schema & helpers
    ├── ingestion.py        # MFAPI + Yahoo Finance live ingestion + risk/returns
    ├── cleaning.py         # Data normalisation utilities
    ├── benchmark.py        # Nifty 50 fetch & returns
    ├── holdings.py         # Holdings analysis & overlap matrix
    ├── flows.py            # Fund flow helpers
    └── risk.py             # Risk metric queries
```

---

## 📡 Data Sources

| Source | What it provides | Endpoint |
|--------|-----------------|----------|
| **MFAPI** | Live NAV, full NAV history, scheme metadata | `https://api.mfapi.in/mf/{scheme_code}` |
| **Yahoo Finance** | Nifty 50 daily price history (benchmark) | `yfinance ^NSEI` |
| **AMFI** | Fund house, category, launch date via MFAPI meta | Embedded in MFAPI response |

---

## 🗄 Database Schema

SQLite (`data/portfolio.db`) with WAL mode:

- **funds** — scheme master with all computed returns, risk metrics, metadata
- **nav_history** — full daily NAV time series per scheme
- **benchmark** — Nifty 50 OHLCV history
- **holdings** — AMC monthly portfolio disclosures
- **flows** — monthly net flow proxy per scheme
- **risk_metrics** — annualised risk metrics (std dev, beta, alpha, Sharpe, Sortino)
- **portfolio_changes** — stock additions, exits, position changes
- **data_quality** — ingestion status per entity

---

## 🚀 Dashboard Tabs

| Tab | Features |
|-----|----------|
| 🏠 Overview | KPI cards, return comparison bar, category pie, NAV cards |
| 📋 Comparison Table | All schemes + **PORTFOLIO AVERAGE** row, sort/filter, CSV export |
| 🗂 Portfolio Analysis | Sector pie, market cap bar, top-10 table, treemap |
| 💸 Fund Flows | Multi-scheme trend line, per-scheme bar, latest net flow |
| 📈 Stock Movements | New additions / exits / increased / reduced positions |
| 🔗 Overlap Analysis | Pairwise overlap heatmap, most/least similar pairs |
| 📉 Benchmark | Nifty 50 chart, fund vs benchmark bars, normalised chart |
| ⚠️ Risk Analysis | Risk-return scatter, volatility bar, full risk table |
| 🔍 Data Quality | Source status, ingestion logs |

---

## ⚙️ Local Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/mutual-fund-portfolio-analytics.git
cd mutual-fund-portfolio-analytics

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The first time the dashboard loads, click **🔄 Refresh All Data** in the sidebar to fetch live data.

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub (make sure `data/portfolio.db` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, entry point `app.py`
4. Click **Deploy**

> **Note:** On Streamlit Cloud the SQLite DB is ephemeral (resets on each deploy). Data is re-fetched automatically on startup via the Refresh button or on first load.

---

## 🔄 Auto-Refresh

Enable **Auto-refresh (60s)** toggle in the sidebar. The dashboard:
1. Waits 60 seconds
2. Clears `st.cache_data`
3. Re-runs the page — fetching updated NAV from MFAPI if you also click **Refresh All Data**

For scheduled background ingestion in a persistent deployment, wire `ingest_schemes(URLS)` into a cron job or APScheduler.

---

## 📈 Computed Metrics

All metrics are calculated from live NAV series — no hardcoded values:

| Metric | Method |
|--------|--------|
| Point-to-Point Returns (1M/3M/6M/1Y) | `(NAV_end / NAV_start - 1) × 100` |
| CAGR (3Y/5Y) | `(NAV_end / NAV_start)^(1/years) - 1` |
| Since Inception CAGR | Same, from first available NAV |
| Standard Deviation | Annualised daily return std dev × √252 |
| Beta | `Cov(fund, Nifty) / Var(Nifty)` |
| Alpha | `Ann_fund_return - β × Ann_bench_return` (excess over risk-free) |
| Sharpe Ratio | `(mean_excess_return / std_dev) × √252` |
| Sortino Ratio | `(mean_excess_return / downside_std) × √252` |

Risk-free rate assumed: **6.5% p.a.** (approximate RBI repo / 10Y G-Sec yield)

---

## 📋 Schemes Covered

| Scheme | Code | Category |
|--------|------|----------|
| Quant Large Cap Fund - Direct | 42367 | Large Cap |
| Kotak Multicap Fund - Direct | 41707 | Multi Cap |
| Canara Robeco Multi Cap - Regular | 43569 | Multi Cap |
| SBI Infrastructure Fund - Direct | 17161 | Sectoral/Thematic |
| ICICI Prudential Focused Equity - Direct | 17412 | Focused Fund |
| Invesco India Focused Fund - Direct | 41096 | Focused Fund |
| ICICI Prudential Dividend Yield - Direct | 26271 | Dividend Yield |

---

## ⚠️ Disclaimer

This platform is for informational and analytical purposes only. It is not investment advice. Mutual fund investments are subject to market risks.
