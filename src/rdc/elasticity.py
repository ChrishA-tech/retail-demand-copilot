"""Price elasticity of weekly demand, estimated as an association (not a causal effect).

Model: log(units) = beta * log(list_price) + product FE + week FE + e, cluster-robust SE by product.
list_price is the median price on lines of 12 units or fewer. The volume-weighted price (revenue / units) is also fitted to show its mechanical bias: this retailer gives bulk
discounts, so big orders lower the average price paid and fake a strong negative elasticity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _fe_fit(d: pd.DataFrame, price_col: str):
    d = d[(d["units"] > 0) & (d[price_col] > 0)].copy()
    d["ly"] = np.log(d["units"])
    d["lp"] = np.log(d[price_col])
    # within transformation for product FE, week dummies for time FE
    d["ly"] -= d.groupby("StockCode")["ly"].transform("mean")
    d["lp"] -= d.groupby("StockCode")["lp"].transform("mean")
    d["wk"] = d["week"].astype(str)
    m = smf.ols("ly ~ lp + C(wk) - 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["StockCode"]})
    return m, len(d)


def pooled(panel: pd.DataFrame) -> dict:
    out = {}
    for name, col in [("list_price", "list_price"), ("median_line_price", "median_price"),
                      ("volume_weighted_price", "avg_paid")]:
        p = panel.assign(avg_paid=panel["revenue"] / panel["units"].replace(0, np.nan))
        m, n = _fe_fit(p, col)
        lo, hi = m.conf_int().loc["lp"]
        out[name] = {"elasticity": round(float(m.params["lp"]), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                     "p_value": float(m.pvalues["lp"]), "n_obs": int(n)}
    return out


def per_product(panel: pd.DataFrame, n_boot: int = 500, seed: int = 0, min_price_changes: int = 4) -> pd.DataFrame:
    """Per-product log-log slope with a block bootstrap CI (resampling weeks). Products with too little price
    variation are marked as not estimable instead of reporting a noisy number."""
    rng = np.random.default_rng(seed)
    rows = []
    for code, g in panel.groupby("StockCode"):
        g = g[(g["units"] > 0) & (g["list_price"] > 0)]
        x, y = np.log(g["list_price"].to_numpy()), np.log(g["units"].to_numpy())
        changes = int((np.abs(np.diff(g["list_price"].to_numpy())) > 1e-9).sum())
        base = {"StockCode": code, "description": g["description"].iat[0] if len(g) else "", "weeks": len(g),
                "price_changes": changes}
        if changes < min_price_changes or np.std(x) < 1e-6:
            rows.append({**base, "elasticity": np.nan, "ci_low": np.nan, "ci_high": np.nan, "estimable": False})
            continue
        slope = np.polyfit(x, y, 1)[0]
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            if np.std(x[i]) > 1e-6:
                boots.append(np.polyfit(x[i], y[i], 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({**base, "elasticity": round(slope, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "estimable": True})
    return pd.DataFrame(rows)
