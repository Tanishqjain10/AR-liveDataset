"""
Mutual Fund Portfolio Analytics Platform
Production-grade Streamlit dashboard.
Data sources: MFAPI (NAV), Yahoo Finance (Benchmark), AMFI (Fund info)
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modules.benchmark import (
    compute_benchmark_returns
    get_benchmark_df
    get_flow_trend
    compute_overlap_matrix
)
from modules.database import init_db, read_sql
from modules.flows import get_flow_trend, get_flows, get_flows_for_scheme
from modules.holdings import (
    compute_overlap_matrix,
    get_all_holdings_latest,
    get_holdings_for_scheme,
)
from modules.ingestion import ingest_schemes
from modules.risk import get_nav_series_from_db, risk_table

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MF Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

URLS = [
    "https://www.valueresearchonline.com/funds/42367/quant-large-cap-fund-direct-plan/",
    "https://www.valueresearchonline.com/funds/41707/kotak-multicap-fund-direct-plan/#risk",
    "https://www.valueresearchonline.com/funds/41707/kotak-multicap-fund-direct-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/41707/kotak-multicap-fund-direct-plan/#other",
    "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/",
    "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/#performance",
    "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/#risk",
    "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/#other",
    "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/",
    "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/#performance",
    "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/#risk",
    "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/#other",
    "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/",
    "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/#performance",
    "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/#risk",
    "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/#other",
    "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/",
    "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/#performance",
    "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/#risk",
    "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/#other",
    "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/",
    "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/#performance",
    "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/#risk",
    "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/#fund-portfolio",
    "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/#other",
]

PALETTE = px.colors.qualitative.Set2

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{decimals}f}%"


def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def color_return(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return "color: #16a34a" if val > 0 else "color: #dc2626"


@st.cache_data(ttl=60)
def load_funds() -> pd.DataFrame:
    rows = read_sql("SELECT * FROM funds ORDER BY scheme_name")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=60)
def load_benchmark() -> pd.DataFrame:
    return get_benchmark_df()


@st.cache_data(ttl=120)
def load_risk() -> pd.DataFrame:
    return risk_table()


@st.cache_data(ttl=120)
def load_holdings() -> pd.DataFrame:
    return get_all_holdings_latest()


@st.cache_data(ttl=120)
def load_flows() -> pd.DataFrame:
    return get_flows()


@st.cache_data(ttl=300)
def load_nav_history(scheme_code: str) -> pd.Series:
    return get_nav_series_from_db(scheme_code)


# ── Initialise DB ─────────────────────────────────────────────────────────────
init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Single_chart_icon.svg/240px-Single_chart_icon.svg.png",
        width=60,
    )
    st.title("MF Analytics")
    st.markdown("---")

    if st.button("🔄 Refresh All Data", use_container_width=True, type="primary"):
        with st.spinner("Fetching live data from MFAPI & Yahoo Finance…"):
            ingest_schemes(URLS)
            nifty_df = fetch_nifty_history()
            store_nifty_history(nifty_df)
            st.cache_data.clear()
        st.success("Data refreshed!")
        st.rerun()

    st.markdown("---")
    funds_df = load_funds()

    if funds_df.empty:
        st.warning("No data yet. Click **Refresh All Data** to load.")
        st.stop()

    search_q = st.text_input("🔍 Search scheme", "")
    all_names = funds_df["scheme_name"].dropna().tolist()
    selected_names = st.multiselect(
        "Select Schemes",
        all_names,
        default=all_names,
    )

    st.markdown("---")
    last_upd = funds_df["last_updated"].dropna()
    if not last_upd.empty:
        st.caption(f"**Last updated:**\n{last_upd.iloc[0][:19]}")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered_df = funds_df.copy()
if search_q:
    filtered_df = filtered_df[filtered_df["scheme_name"].str.contains(search_q, case=False, na=False)]
if selected_names:
    filtered_df = filtered_df[filtered_df["scheme_name"].isin(selected_names)]

if filtered_df.empty:
    st.warning("No schemes match the current filter.")
    st.stop()

# ── Auto-refresh logic ────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ── Navigation tabs ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {background:#f8fafc;border-radius:10px;padding:16px 20px;border:1px solid #e2e8f0}
    .stTabs [data-baseweb="tab-list"] {gap:6px}
    .stTabs [data-baseweb="tab"] {border-radius:8px 8px 0 0;padding:8px 20px;font-weight:600}
    thead tr th {background:#1e3a5f !important;color:white !important}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Mutual Fund Portfolio Analytics Platform")
st.caption("Live data: MFAPI · Yahoo Finance · AMFI  |  Auto-updates every 60s")

(
    tab_overview,
    tab_comparison,
    tab_portfolio,
    tab_flows,
    tab_stock_moves,
    tab_overlap,
    tab_benchmark,
    tab_risk,
    tab_quality,
) = st.tabs([
    "🏠 Overview",
    "📋 Comparison Table",
    "🗂 Portfolio Analysis",
    "💸 Fund Flows",
    "📈 Stock Movements",
    "🔗 Overlap Analysis",
    "📉 Benchmark",
    "⚠️ Risk Analysis",
    "🔍 Data Quality",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Schemes", len(filtered_df))
    live_count = int((filtered_df["source_status"] == "Live").sum())
    c2.metric("Live Data Feeds", live_count)
    avg_1y = filtered_df["one_year_return"].mean()
    c3.metric("Avg 1Y Return", fmt_pct(avg_1y) if not np.isnan(avg_1y) else "—")
    avg_3y = filtered_df["three_year_cagr"].mean()
    c4.metric("Avg 3Y CAGR", fmt_pct(avg_3y) if not np.isnan(avg_3y) else "—")
    avg_beta = filtered_df["beta"].mean()
    c5.metric("Avg Beta", fmt_num(avg_beta) if not np.isnan(avg_beta) else "—")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Return Comparison (1Y vs 3Y CAGR)")
        plot_df = filtered_df[["scheme_name", "one_year_return", "three_year_cagr"]].dropna(subset=["scheme_name"])
        plot_df = plot_df.rename(columns={"one_year_return": "1Y Return", "three_year_cagr": "3Y CAGR"})
        melted = plot_df.melt(id_vars="scheme_name", var_name="Period", value_name="Return (%)")
        if not melted.empty:
            fig = px.bar(
                melted, x="scheme_name", y="Return (%)", color="Period",
                barmode="group", color_discrete_sequence=PALETTE,
                labels={"scheme_name": ""},
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=80))
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Category Breakdown")
        cat_counts = filtered_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        if not cat_counts.empty:
            fig2 = px.pie(
                cat_counts, names="Category", values="Count",
                color_discrete_sequence=PALETTE, hole=0.45,
            )
            fig2.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    # Latest NAV cards
    st.subheader("Latest NAV")
    nav_cols = st.columns(min(len(filtered_df), 4))
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        if i >= 4:
            break
        with nav_cols[i]:
            nav_val = f"₹{row['nav']:.4f}" if row["nav"] else "—"
            delta = fmt_pct(row["one_month_return"])
            st.metric(row["scheme_name"][:28], nav_val, delta, delta_color="normal")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
with tab_comparison:
    st.subheader("Scheme Comparison Table")

    DISPLAY_COLS = {
        "scheme_name":          "Scheme Name",
        "category":             "Category",
        "aum":                  "AUM (Cr)",
        "expense_ratio":        "Exp. Ratio",
        "since_inception_return": "Since Incep.",
        "one_month_return":     "1M",
        "three_month_return":   "3M",
        "six_month_return":     "6M",
        "one_year_return":      "1Y",
        "three_year_cagr":      "3Y CAGR",
        "five_year_cagr":       "5Y CAGR",
        "std_dev":              "Std Dev",
        "beta":                 "Beta",
        "sharpe_ratio":         "Sharpe",
        "sortino_ratio":        "Sortino",
        "category_rank":        "Cat. Rank",
        "holdings_count":       "# Stocks",
        "fund_manager":         "Fund Manager",
        "launch_date":          "Launch Date",
    }

    display_df = filtered_df[list(DISPLAY_COLS.keys())].copy()
    display_df = display_df.rename(columns=DISPLAY_COLS)

    # Compute PORTFOLIO AVERAGE row
    num_cols = [
        "AUM (Cr)", "Exp. Ratio", "Since Incep.", "1M", "3M", "6M", "1Y",
        "3Y CAGR", "5Y CAGR", "Std Dev", "Beta", "Sharpe", "Sortino",
        "Cat. Rank", "# Stocks"
    ]
    avg_row = {"Scheme Name": "📊 PORTFOLIO AVERAGE"}
    for c in display_df.columns:
        if c in num_cols:
            avg_row[c] = display_df[c].mean(skipna=True)
        elif c not in avg_row:
            avg_row[c] = "—"

    avg_df = pd.DataFrame([avg_row])
    full_df = pd.concat([display_df, avg_df], ignore_index=True)

    # Highlight average row
    def highlight_avg(row):
        if row["Scheme Name"] == "📊 PORTFOLIO AVERAGE":
            return ["background-color: #1e3a5f; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    pct_cols = ["Since Incep.", "1M", "3M", "6M", "1Y", "3Y CAGR", "5Y CAGR", "Std Dev"]
    formatted = full_df.copy()
    for c in pct_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].apply(lambda v: fmt_pct(v) if isinstance(v, (int, float)) else v)

    st.dataframe(
        full_df.style.apply(highlight_avg, axis=1),
        use_container_width=True,
        height=420,
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        sort_col = st.selectbox("Sort by", options=list(DISPLAY_COLS.values()), index=8)
    with cc2:
        sort_asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"

    orig_col = {v: k for k, v in DISPLAY_COLS.items()}.get(sort_col, sort_col)
    if orig_col in filtered_df.columns:
        sorted_df = filtered_df.sort_values(orig_col, ascending=sort_asc, na_position="last")
        display_sorted = sorted_df[list(DISPLAY_COLS.keys())].rename(columns=DISPLAY_COLS)
        st.dataframe(display_sorted, use_container_width=True, height=300)

    csv_data = full_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV", csv_data, "scheme_comparison.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.subheader("Portfolio Analysis")
    scheme_options = filtered_df["scheme_name"].tolist()
    sel_scheme = st.selectbox("Select Scheme", scheme_options)
    sel_code = filtered_df[filtered_df["scheme_name"] == sel_scheme]["scheme_code"].iloc[0]

    h_df = get_holdings_for_scheme(sel_code)

    if h_df.empty:
        st.info("Holdings data not yet available for this scheme. Click **Refresh All Data** to fetch latest AMC disclosures.")
    else:
        latest_date = h_df["as_of_date"].max()
        st.caption(f"Portfolio as of: **{latest_date}**")
        latest_h = h_df[h_df["as_of_date"] == latest_date]

        pa_col1, pa_col2, pa_col3 = st.columns(3)
        total_invested = latest_h["weight"].sum()
        cash_pct = max(0, 100 - total_invested)
        pa_col1.metric("# Holdings", len(latest_h))
        pa_col2.metric("Invested %", f"{total_invested:.1f}%")
        pa_col3.metric("Cash/Others", f"{cash_pct:.1f}%")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Sector Allocation**")
            if "sector" in latest_h.columns:
                sec_df = latest_h.groupby("sector")["weight"].sum().reset_index().sort_values("weight", ascending=False)
                if cash_pct > 0:
                    sec_df = pd.concat([sec_df, pd.DataFrame([{"sector": "Cash/Others", "weight": cash_pct}])], ignore_index=True)
                fig_sec = px.pie(sec_df, names="sector", values="weight", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_sec.update_layout(height=350, showlegend=True, margin=dict(t=10, b=0))
                st.plotly_chart(fig_sec, use_container_width=True)

        with p2:
            st.markdown("**Market Cap Allocation**")
            if "market_cap_bucket" in latest_h.columns:
                mc_df = latest_h.groupby("market_cap_bucket")["weight"].sum().reset_index().sort_values("weight", ascending=False)
                fig_mc = px.bar(mc_df, x="market_cap_bucket", y="weight", color="market_cap_bucket",
                                color_discrete_sequence=PALETTE, labels={"weight": "% NAV", "market_cap_bucket": ""})
                fig_mc.update_layout(height=350, showlegend=False, margin=dict(t=10, b=0))
                st.plotly_chart(fig_mc, use_container_width=True)

        with p3:
            st.markdown("**Top 10 Holdings**")
            top10 = latest_h.nlargest(10, "weight")[["holding_name", "sector", "weight"]].reset_index(drop=True)
            top10.columns = ["Stock", "Sector", "% NAV"]
            st.dataframe(top10, use_container_width=True, height=350, hide_index=True)

        st.markdown("**Top Holdings Treemap**")
        if len(latest_h) > 0:
            treemap_df = latest_h.nlargest(20, "weight").copy()
            treemap_df["holding_name"] = treemap_df["holding_name"].fillna("Unknown")
            treemap_df["sector"] = treemap_df["sector"].fillna("Other")
            fig_tree = px.treemap(
                treemap_df, path=["sector", "holding_name"],
                values="weight", color="weight",
                color_continuous_scale="Blues",
                labels={"weight": "% NAV"},
            )
            fig_tree.update_layout(height=420, margin=dict(t=10, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("**Full Holdings Table**")
        st.dataframe(latest_h[["holding_name", "isin", "sector", "market_cap_bucket", "weight", "value"]].rename(
            columns={"holding_name": "Stock", "isin": "ISIN", "sector": "Sector",
                     "market_cap_bucket": "Market Cap", "weight": "% NAV", "value": "Value (Cr)"}
        ), use_container_width=True, height=300, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — FUND FLOWS
# ═══════════════════════════════════════════════════════════════════
with tab_flows:
    st.subheader("Fund Flow Analysis")
    st.caption("Monthly NAV-derived net flow proxy (positive = positive momentum month, negative = drawdown month). Actual AMFI inflow/outflow data requires institutional data access.")

    flows_df = load_flows()

    if flows_df.empty:
        st.info("No flow data yet. Click **Refresh All Data**.")
    else:
        # Merge scheme names
        name_map = dict(zip(filtered_df["scheme_code"], filtered_df["scheme_name"]))

        # Multi-scheme flow trend
        fl_col1, fl_col2 = st.columns([3, 2])
        with fl_col1:
            st.markdown("**Monthly Flow Trend (All Schemes)**")
            trend_frames = []
            for code in filtered_df["scheme_code"]:
                t = get_flow_trend(code, 24)
                if not t.empty:
                    t["Scheme"] = name_map.get(code, code)
                    trend_frames.append(t)
            if trend_frames:
                all_trends = pd.concat(trend_frames, ignore_index=True)
                fig_flow = px.line(
                    all_trends, x="period", y="amount", color="Scheme",
                    labels={"amount": "Net Flow Index", "period": "Month"},
                    color_discrete_sequence=PALETTE,
                )
                fig_flow.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=60))
                fig_flow.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_flow, use_container_width=True)

        with fl_col2:
            st.markdown("**Latest Net Flow by Scheme**")
            latest_flows = []
            for code in filtered_df["scheme_code"]:
                t = get_flow_trend(code, 1)
                if not t.empty:
                    latest_flows.append({
                        "Scheme": name_map.get(code, code)[:25],
                        "Net Flow": round(float(t["amount"].iloc[-1]), 4),
                    })
            if latest_flows:
                lf_df = pd.DataFrame(latest_flows)
                lf_df["Color"] = lf_df["Net Flow"].apply(lambda v: "Positive" if v >= 0 else "Negative")
                fig_lf = px.bar(
                    lf_df, x="Net Flow", y="Scheme", orientation="h",
                    color="Color", color_discrete_map={"Positive": "#16a34a", "Negative": "#dc2626"},
                )
                fig_lf.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_lf, use_container_width=True)

        # Per-scheme detail
        st.markdown("---")
        st.markdown("**Per-Scheme Flow History**")
        sel_flow_scheme = st.selectbox("Scheme", filtered_df["scheme_name"].tolist(), key="flow_sel")
        flow_code = filtered_df[filtered_df["scheme_name"] == sel_flow_scheme]["scheme_code"].iloc[0]
        per_scheme_flows = get_flow_trend(flow_code, 36)
        if not per_scheme_flows.empty:
            per_scheme_flows["Direction"] = per_scheme_flows["amount"].apply(lambda v: "Positive" if v >= 0 else "Negative")
            fig_ps = px.bar(
                per_scheme_flows, x="period", y="amount", color="Direction",
                color_discrete_map={"Positive": "#16a34a", "Negative": "#dc2626"},
                labels={"amount": "Net Flow Index", "period": "Month"},
            )
            fig_ps.update_layout(height=350, showlegend=False, margin=dict(t=20, b=60))
            fig_ps.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_ps, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — STOCK MOVEMENTS
# ═══════════════════════════════════════════════════════════════════
with tab_stock_moves:
    st.subheader("Stock Movement Analysis")
    st.caption("Compares current vs previous month holdings from AMC disclosures.")

    moves_rows = read_sql("SELECT * FROM portfolio_changes ORDER BY created_at DESC")
    moves_df = pd.DataFrame([dict(r) for r in moves_rows]) if moves_rows else pd.DataFrame()

    if moves_df.empty:
        st.info("Portfolio change data populates once two consecutive months of holdings are available.")
    else:
        name_map = dict(zip(filtered_df["scheme_code"], filtered_df["scheme_name"]))
        moves_df["scheme_label"] = moves_df["scheme_code"].map(name_map).fillna(moves_df["scheme_code"])

        sel_move_scheme = st.selectbox("Scheme", moves_df["scheme_label"].unique().tolist(), key="moves_sel")
        scheme_moves = moves_df[moves_df["scheme_label"] == sel_move_scheme]

        for ctype, icon in [("NEW_ADDITION", "🟢"), ("INCREASED", "⬆️"), ("REDUCED", "⬇️"), ("COMPLETE_EXIT", "🔴")]:
            sub = scheme_moves[scheme_moves["change_type"] == ctype]
            if not sub.empty:
                st.markdown(f"**{icon} {ctype.replace('_', ' ')} ({len(sub)})**")
                disp = sub[["holding_name", "isin", "previous_weight", "current_weight", "weight_change"]].rename(
                    columns={
                        "holding_name": "Stock", "isin": "ISIN",
                        "previous_weight": "Prev %", "current_weight": "Curr %",
                        "weight_change": "Δ Weight",
                    }
                )
                st.dataframe(disp, use_container_width=True, hide_index=True)

        # Summary bar
        summary = scheme_moves["change_type"].value_counts().reset_index()
        summary.columns = ["Change Type", "Count"]
        fig_sum = px.bar(
            summary, x="Change Type", y="Count",
            color="Change Type",
            color_discrete_map={
                "NEW_ADDITION": "#16a34a", "INCREASED": "#22c55e",
                "REDUCED": "#f97316", "COMPLETE_EXIT": "#dc2626",
            },
        )
        fig_sum.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_sum, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 6 — OVERLAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_overlap:
    st.subheader("Portfolio Overlap Analysis")

    h_all = load_holdings()
    if h_all.empty:
        st.info("Holdings data required for overlap analysis. Refresh data first.")
    else:
        # Filter to selected schemes only
        sel_codes = filtered_df["scheme_code"].tolist()
        h_sel = h_all[h_all["scheme_code"].isin(sel_codes)].copy()
        # Attach scheme names
        name_map = dict(zip(filtered_df["scheme_code"], filtered_df["scheme_name"]))
        h_sel["scheme_name"] = h_sel["scheme_code"].map(name_map)

        overlap_mat = compute_overlap_matrix(h_sel)

        if overlap_mat.empty:
            st.info("Not enough holdings data to compute overlap.")
        else:
            st.markdown("**Overlap Heatmap (% portfolio overlap)**")
            fig_heat = px.imshow(
                overlap_mat,
                text_auto=".1f",
                color_continuous_scale="Blues",
                zmin=0, zmax=100,
                labels={"color": "Overlap %"},
            )
            fig_heat.update_layout(height=480, margin=dict(t=20, b=80, l=120, r=20))
            st.plotly_chart(fig_heat, use_container_width=True)

            # Most / least similar
            ol_vals = overlap_mat.copy()
            np.fill_diagonal(ol_vals.values, np.nan)
            ov_stacked = ol_vals.stack().reset_index()
            ov_stacked.columns = ["Fund A", "Fund B", "Overlap %"]
            ov_sorted = ov_stacked.dropna().sort_values("Overlap %", ascending=False)
            ov_sorted = ov_sorted[ov_sorted["Fund A"] < ov_sorted["Fund B"]]  # deduplicate pairs

            o1, o2 = st.columns(2)
            with o1:
                st.markdown("**Most Overlapping Pairs**")
                st.dataframe(ov_sorted.head(5).reset_index(drop=True), use_container_width=True, hide_index=True)
            with o2:
                st.markdown("**Least Overlapping Pairs**")
                st.dataframe(ov_sorted.tail(5).reset_index(drop=True), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 7 — BENCHMARK
# ═══════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.subheader("Benchmark Analysis — Nifty 50")

    bench_df = load_benchmark()

    if bench_df.empty:
        st.info("Benchmark data not loaded. Click **Refresh All Data**.")
    else:
        bench_rets = compute_benchmark_returns(bench_df)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Nifty 1M",  fmt_pct(bench_rets.get("1M")))
        b2.metric("Nifty 3M",  fmt_pct(bench_rets.get("3M")))
        b3.metric("Nifty 1Y",  fmt_pct(bench_rets.get("1Y")))
        b4.metric("Nifty 3Y CAGR", fmt_pct(bench_rets.get("3Y")))
        b5.metric("Nifty 5Y CAGR", fmt_pct(bench_rets.get("5Y")))

        st.markdown("---")

        # Nifty chart
        bench_plot = bench_df.reset_index()
        bench_plot.columns = ["date", "close"]
        bench_plot = bench_plot.dropna()
        bench_plot = bench_plot[bench_plot["date"] >= pd.Timestamp("2015-01-01")]

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=bench_plot["date"], y=bench_plot["close"],
            mode="lines", name="Nifty 50",
            line=dict(color="#1e3a5f", width=2),
            fill="tozeroy", fillcolor="rgba(30,58,95,0.08)",
        ))
        fig_bench.update_layout(
            title="Nifty 50 — Historical Close Price",
            xaxis_title="Date", yaxis_title="Index Level",
            height=420, margin=dict(l=0, r=0, t=50, b=40),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        # Fund vs Benchmark comparison
        st.markdown("**Fund Returns vs Nifty 50**")
        periods_compare = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]
        bench_map = {
            "1M": bench_rets.get("1M"), "3M": bench_rets.get("3M"),
            "6M": bench_rets.get("6M"), "1Y": bench_rets.get("1Y"),
            "3Y": bench_rets.get("3Y"), "5Y": bench_rets.get("5Y"),
        }
        fund_ret_map = {
            "1M": "one_month_return", "3M": "three_month_return",
            "6M": "six_month_return", "1Y": "one_year_return",
            "3Y": "three_year_cagr", "5Y": "five_year_cagr",
        }

        for _, fund_row in filtered_df.iterrows():
            fund_vals = [fund_row.get(fund_ret_map[p]) for p in periods_compare]
            bench_vals = [bench_map.get(p) for p in periods_compare]
            alpha_vals = [
                round(f - b, 2) if f is not None and b is not None else None
                for f, b in zip(fund_vals, bench_vals)
            ]

            fig_fb = go.Figure()
            fig_fb.add_trace(go.Bar(name="Fund", x=periods_compare, y=fund_vals,
                                    marker_color="#1e3a5f"))
            fig_fb.add_trace(go.Bar(name="Nifty 50", x=periods_compare, y=bench_vals,
                                    marker_color="#94a3b8"))
            fig_fb.update_layout(
                title=fund_row["scheme_name"],
                barmode="group", height=300,
                margin=dict(l=0, r=0, t=40, b=20),
                yaxis_title="Return (%)",
            )
            with st.expander(fund_row["scheme_name"], expanded=False):
                st.plotly_chart(fig_fb, use_container_width=True)
                alpha_row = pd.DataFrame([{
                    "Period": p,
                    "Fund %": fmt_pct(f),
                    "Nifty %": fmt_pct(b),
                    "Alpha": fmt_pct(a),
                } for p, f, b, a in zip(periods_compare, fund_vals, bench_vals, alpha_vals)])
                st.dataframe(alpha_row, use_container_width=True, hide_index=True)

        # Normalised NAV vs Nifty chart
        st.markdown("**Normalised Performance (Base=100)**")
        bench_norm = bench_df["close"].dropna()
        bench_norm = bench_norm[bench_norm.index >= pd.Timestamp("2020-01-01")]
        if not bench_norm.empty:
            bench_norm = bench_norm / bench_norm.iloc[0] * 100
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(
                x=bench_norm.index, y=bench_norm.values,
                name="Nifty 50", line=dict(color="black", dash="dash", width=2),
            ))
            for i, (_, fund_row) in enumerate(filtered_df.iterrows()):
                nav_s = load_nav_history(fund_row["scheme_code"])
                if nav_s.empty:
                    continue
                nav_s = nav_s[nav_s.index >= pd.Timestamp("2020-01-01")]
                if nav_s.empty:
                    continue
                nav_norm = nav_s / nav_s.iloc[0] * 100
                fig_norm.add_trace(go.Scatter(
                    x=nav_norm.index, y=nav_norm.values,
                    name=fund_row["scheme_name"][:25],
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                ))
            fig_norm.update_layout(
                height=500, xaxis_title="Date", yaxis_title="Normalised Value (Base=100)",
                margin=dict(l=0, r=0, t=20, b=40),
            )
            st.plotly_chart(fig_norm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 8 — RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk Analysis")

    risk_df = load_risk()
    if risk_df.empty:
        st.info("Risk data not available yet. Refresh data.")
    else:
        # Filter to selected
        risk_filt = risk_df[risk_df["scheme_code"].isin(filtered_df["scheme_code"].tolist())]
        name_map = dict(zip(filtered_df["scheme_code"], filtered_df["scheme_name"]))
        risk_filt = risk_filt.copy()
        risk_filt["scheme_label"] = risk_filt["scheme_code"].map(name_map).fillna(risk_filt["scheme_code"])

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Risk-Return Scatter**")
            scatter_df = filtered_df[["scheme_name", "one_year_return", "std_dev", "beta"]].dropna(
                subset=["one_year_return", "std_dev"]
            )
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="std_dev", y="one_year_return",
                    text="scheme_name",
                    size=scatter_df["beta"].abs().fillna(1) * 30,
                    color="one_year_return",
                    color_continuous_scale="RdYlGn",
                    labels={"std_dev": "Std Dev (Risk)", "one_year_return": "1Y Return (%)"},
                )
                fig_scatter.update_traces(textposition="top center", textfont_size=10)
                fig_scatter.update_layout(height=420, showlegend=False, margin=dict(t=20))
                st.plotly_chart(fig_scatter, use_container_width=True)

        with r2:
            st.markdown("**Risk Ranking (Lower Std Dev = Better)**")
            rank_df = filtered_df[["scheme_name", "std_dev", "beta", "sharpe_ratio", "sortino_ratio"]].copy()
            rank_df = rank_df.sort_values("std_dev", na_position="last")
            rank_df.columns = ["Scheme", "Std Dev", "Beta", "Sharpe", "Sortino"]
            st.dataframe(rank_df.reset_index(drop=True), use_container_width=True, height=420, hide_index=True)

        st.markdown("---")

        # Volatility bar chart
        st.markdown("**Annualised Volatility Comparison**")
        vol_df = risk_filt[["scheme_label", "volatility"]].dropna().sort_values("volatility", ascending=True)
        if not vol_df.empty:
            fig_vol = px.bar(
                vol_df, x="volatility", y="scheme_label", orientation="h",
                color="volatility", color_continuous_scale="OrRd",
                labels={"volatility": "Volatility (%)", "scheme_label": ""},
            )
            fig_vol.update_layout(height=350, margin=dict(l=200, r=20, t=20, b=40), showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        # Full risk table
        st.markdown("**Complete Risk Metrics Table**")
        risk_display = risk_filt[[
            "scheme_label", "standard_deviation", "beta", "sharpe_ratio",
            "sortino_ratio", "alpha", "volatility", "downside_volatility"
        ]].rename(columns={
            "scheme_label": "Scheme",
            "standard_deviation": "Std Dev",
            "beta": "Beta",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "alpha": "Alpha",
            "volatility": "Volatility",
            "downside_volatility": "Downside Vol",
        })
        st.dataframe(risk_display.reset_index(drop=True), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 9 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════
with tab_quality:
    st.subheader("Data Quality Dashboard")

    dq_rows = read_sql("SELECT * FROM data_quality ORDER BY last_updated DESC")
    dq_df = pd.DataFrame([dict(r) for r in dq_rows]) if dq_rows else pd.DataFrame()

    if dq_df.empty:
        st.info("No data quality records yet.")
    else:
        ok_count = int((dq_df["status"] == "OK").sum())
        err_count = int((dq_df["status"] == "ERROR").sum())
        missing_count = int(dq_df["status"].str.contains("MISSING", na=False).sum())

        dq_c1, dq_c2, dq_c3 = st.columns(3)
        dq_c1.metric("✅ OK", ok_count)
        dq_c2.metric("❌ Errors", err_count)
        dq_c3.metric("⚠️ Missing", missing_count)

        st.dataframe(
            dq_df[["entity_type", "entity_key", "metric_name", "source", "status", "notes", "last_updated"]],
            use_container_width=True, height=400,
        )

    # Source status summary from funds table
    st.markdown("**Live Source Status**")
    src_df = filtered_df[["scheme_name", "source_status", "last_updated", "source_primary"]].copy()
    src_df.columns = ["Scheme", "Status", "Last Updated", "Primary Source"]

    def status_icon(s):
        if s == "Live":
            return "🟢 Live"
        elif s == "Unavailable":
            return "🔴 Unavailable"
        return f"🟡 {s}"

    src_df["Status"] = src_df["Status"].apply(status_icon)
    st.dataframe(src_df.reset_index(drop=True), use_container_width=True, hide_index=True)
