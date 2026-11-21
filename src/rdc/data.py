"""Load and clean Online Retail II, then build a weekly product panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = "https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda"
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002",
               "PADS", "D", "CRUK", "S", "B", "m", "gift_0001_10", "gift_0001_20", "gift_0001_30", "gift_0001_40",
               "gift_0001_50"}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".rda":
        import pyreadr
        df = next(iter(pyreadr.read_r(str(path)).values()))
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
        df = df.rename(columns={"Customer ID": "CustomerID"})
    else:
        df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop cancellations, returns, zero prices, non-product codes and exact duplicates. Returns a log of removals."""
    log = {"raw_rows": len(df)}
    steps = [
        ("cancelled_invoices", df["Invoice"].str.upper().str.startswith("C")),
        ("non_positive_quantity", df["Quantity"] <= 0),
        ("non_positive_price", df["Price"] <= 0),
        ("non_product_codes", df["StockCode"].isin(NON_PRODUCT) | df["StockCode"].str.upper().str.startswith("TEST")),
    ]
    keep = pd.Series(True, index=df.index)
    for name, mask in steps:
        log[name] = int((mask & keep).sum())
        keep &= ~mask
    out = df[keep]
    dup = out.duplicated()
    log["exact_duplicates"] = int(dup.sum())
    out = out[~dup].copy()
    log["clean_rows"] = len(out)
    log["removed_pct"] = round(100 * (1 - len(out) / log["raw_rows"]), 2)
    return out, log


def weekly_panel(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Weekly units, revenue, list price (median price on lines of 12 units or fewer, i.e. before bulk discounts),
    median line price and buyers for the top_n products by revenue.

    Weeks start on Monday. Partial first/last weeks are dropped; products must sell in at least half of the weeks; weeks with no sales are filled with zero units
    and the product's last observed median price (carried forward).
    """
    df = df.assign(revenue=df["Quantity"] * df["Price"],
                   week=df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time)
    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")[1:-1]  # drop partial first/last weeks
    df = df[df["week"].isin(weeks)]
    # rank on full weeks only, and require sales in at least half the weeks (a one-off bulk order is not demand)
    active = df.groupby("StockCode")["week"].nunique() >= len(weeks) // 2
    top = df[df["StockCode"].isin(active[active].index)].groupby("StockCode")["revenue"].sum().nlargest(top_n).index
    d = df[df["StockCode"].isin(top)]
    g = d.groupby(["StockCode", "week"])
    small = d[d["Quantity"] <= 12].groupby(["StockCode", "week"])["Price"].median()
    panel = pd.DataFrame({
        "units": g["Quantity"].sum(),
        "revenue": g["revenue"].sum(),
        "median_price": g["Price"].median(),
        "list_price": small,
        "buyers": g["CustomerID"].nunique(),
        "lines": g.size(),
    }).reset_index()
    idx = pd.MultiIndex.from_product([top, weeks], names=["StockCode", "week"])
    panel = panel.set_index(["StockCode", "week"]).reindex(idx).reset_index()
    for c in ["units", "revenue", "buyers", "lines"]:
        panel[c] = panel[c].fillna(0)
    for c in ["median_price", "list_price"]:
        panel[c] = panel.groupby("StockCode")[c].transform(lambda s: s.ffill().bfill())
    panel["list_price"] = panel["list_price"].fillna(panel["median_price"])
    desc = d.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    panel["description"] = panel["StockCode"].map(desc).astype(str).str.strip()
    return panel.sort_values(["StockCode", "week"]).reset_index(drop=True)
