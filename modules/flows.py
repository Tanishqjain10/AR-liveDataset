import pandas as pd

from .database import read_sql


def get_flows() -> pd.DataFrame:
    rows = read_sql("SELECT * FROM flows ORDER BY scheme_code, period DESC")
    return pd.DataFrame([dict(r) for r in rows])


def get_flows_for_scheme(scheme_code: str) -> pd.DataFrame:
    rows = read_sql(
        "SELECT period, flow_type, amount FROM flows WHERE scheme_code=? ORDER BY period",
        [scheme_code],
    )
    return pd.DataFrame([dict(r) for r in rows])


def get_flow_trend(scheme_code: str, periods: int = 12) -> pd.DataFrame:
    rows = read_sql(
        """SELECT period, amount FROM flows
           WHERE scheme_code=? AND flow_type='net_nav_proxy'
           ORDER BY period DESC LIMIT ?""",
        [scheme_code, periods],
    )
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    return df.sort_values("period").reset_index(drop=True)
