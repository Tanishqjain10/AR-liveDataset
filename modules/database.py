import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path("/tmp/portfolio.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS funds (
    scheme_code TEXT PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    category TEXT,
    plan_type TEXT,
    amc TEXT,
    launch_date TEXT,
    fund_manager TEXT,
    aum REAL,
    expense_ratio REAL,
    nav REAL,
    nav_date TEXT,
    since_inception_return REAL,
    one_month_return REAL,
    three_month_return REAL,
    six_month_return REAL,
    one_year_return REAL,
    three_year_cagr REAL,
    five_year_cagr REAL,
    std_dev REAL,
    beta REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    alpha REAL,
    category_rank REAL,
    holdings_count INTEGER,
    benchmark_name TEXT,
    benchmark_symbol TEXT,
    source_url TEXT,
    source_primary TEXT,
    source_secondary TEXT,
    last_updated TEXT,
    source_status TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS nav_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT NOT NULL,
    nav_date TEXT NOT NULL,
    nav REAL,
    UNIQUE(scheme_code, nav_date)
);

CREATE TABLE IF NOT EXISTS benchmark (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_name TEXT NOT NULL,
    benchmark_symbol TEXT,
    source TEXT,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(benchmark_name, date)
);

CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT NOT NULL,
    scheme_name TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    holding_name TEXT NOT NULL,
    isin TEXT,
    sector TEXT,
    market_cap_bucket TEXT,
    weight REAL,
    value REAL,
    shares REAL,
    source_url TEXT,
    source_status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scheme_code, as_of_date, holding_name, isin)
);

CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT,
    scheme_name TEXT,
    category TEXT,
    period TEXT NOT NULL,
    flow_type TEXT NOT NULL,
    amount REAL,
    source TEXT,
    source_url TEXT,
    last_updated TEXT,
    source_status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scheme_code, period, flow_type, source)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT NOT NULL,
    scheme_name TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    standard_deviation REAL,
    beta REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    alpha REAL,
    volatility REAL,
    downside_volatility REAL,
    tracking_error REAL,
    information_ratio REAL,
    source TEXT,
    source_url TEXT,
    source_status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scheme_code, as_of_date)
);

CREATE TABLE IF NOT EXISTS portfolio_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT NOT NULL,
    scheme_name TEXT NOT NULL,
    current_as_of_date TEXT NOT NULL,
    previous_as_of_date TEXT,
    holding_name TEXT NOT NULL,
    isin TEXT,
    change_type TEXT NOT NULL,
    weight_change REAL,
    value_change REAL,
    current_weight REAL,
    previous_weight REAL,
    source_url TEXT,
    source_status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    source TEXT,
    last_updated TEXT,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_key, metric_name)
);
"""


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception as e:
        print("SQLite warning:", e)

    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def upsert(table: str, row: dict[str, Any], conflict_cols: list[str], update_cols: list[str] | None = None):
    if not row:
        return
    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    conflict = ", ".join(conflict_cols)
    if update_cols is None:
        update_cols = [c for c in cols if c not in conflict_cols]
    updates = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    sql = f"""
    INSERT INTO {table} ({", ".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT({conflict}) DO UPDATE SET {updates}
    """
    values = [row[c] for c in cols]
    with get_conn() as conn:
        conn.execute(sql, values)
        conn.commit()


def bulk_upsert(table: str, rows: list[dict[str, Any]], conflict_cols: list[str], update_cols: list[str] | None = None):
    for row in rows:
        upsert(table, row, conflict_cols, update_cols)


def read_sql(query: str, params: Iterable[Any] | None = None):
    with get_conn() as conn:
        cur = conn.execute(query, tuple(params or []))
        rows = cur.fetchall()
    return rows


def execute(sql: str, params: Iterable[Any] | None = None):
    with get_conn() as conn:
        conn.execute(sql, tuple(params or []))
        conn.commit()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def from_json(text: str | None):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None
