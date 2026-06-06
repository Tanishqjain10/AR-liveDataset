import pandas as pd

from .database import read_sql


def get_holdings_for_scheme(scheme_code: str) -> pd.DataFrame:
    rows = read_sql(
        """SELECT scheme_code, scheme_name, as_of_date, holding_name, isin,
                  sector, market_cap_bucket, weight, value, shares
           FROM holdings
           WHERE scheme_code = ?
           ORDER BY as_of_date DESC, weight DESC""",
        [scheme_code],
    )
    return pd.DataFrame([dict(r) for r in rows])


def get_all_holdings_latest() -> pd.DataFrame:
    rows = read_sql(
        """SELECT h.*
           FROM holdings h
           JOIN (SELECT scheme_code, MAX(as_of_date) AS as_of_date
                 FROM holdings GROUP BY scheme_code) x
           ON h.scheme_code = x.scheme_code AND h.as_of_date = x.as_of_date
           ORDER BY h.scheme_code, h.weight DESC""",
    )
    return pd.DataFrame([dict(r) for r in rows])


def compute_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise portfolio overlap matrix.
    overlap(A,B) = sum of min(wA_i, wB_i) for all common holdings.
    """
    if df.empty:
        return pd.DataFrame()
    schemes = df["scheme_code"].unique().tolist()
    name_map = {}
    for _, r in df.iterrows():
        name_map[r["scheme_code"]] = r.get("scheme_name", r["scheme_code"])

    weights = {}
    for code in schemes:
        sub = df[df["scheme_code"] == code]
        key_col = "isin" if "isin" in sub.columns else "holding_name"
        weights[code] = dict(zip(sub[key_col].fillna(sub["holding_name"]), sub["weight"].fillna(0)))

    labels = [name_map.get(c, c) for c in schemes]
    import numpy as np
    n = len(schemes)
    mat = np.zeros((n, n))
    for i, ci in enumerate(schemes):
        for j, cj in enumerate(schemes):
            if i == j:
                mat[i, j] = 100.0
                continue
            common = set(weights[ci]) & set(weights[cj])
            mat[i, j] = round(sum(min(weights[ci][k], weights[cj][k]) for k in common), 2)
    return pd.DataFrame(mat, index=labels, columns=labels)
