import numpy as np
import pandas as pd

from .database import read_sql


def risk_table() -> pd.DataFrame:
    rows = read_sql(
        """SELECT rm.scheme_code, f.scheme_name, rm.as_of_date,
                  rm.standard_deviation, rm.beta, rm.sharpe_ratio,
                  rm.sortino_ratio, rm.alpha, rm.volatility,
                  rm.downside_volatility, rm.source_status
           FROM risk_metrics rm
           JOIN funds f ON f.scheme_code = rm.scheme_code
           ORDER BY rm.as_of_date DESC"""
    )
    return pd.DataFrame([dict(r) for r in rows])


def get_nav_series_from_db(scheme_code: str) -> pd.Series:
    rows = read_sql(
        "SELECT nav_date, nav FROM nav_history WHERE scheme_code=? ORDER BY nav_date",
        [scheme_code]
    )
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame([dict(r) for r in rows])
    df["nav_date"] = pd.to_datetime(df["nav_date"])
    return df.set_index("nav_date")["nav"].sort_index()
