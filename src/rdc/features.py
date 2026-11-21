"""Leak-free features: every value for week t uses only data up to week t-1."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4, 8, 13, 52]
FEATURES = [f"lag_{k}" for k in LAGS] + ["roll_mean_4", "roll_std_4", "roll_mean_13", "level", "zero_share_13",
            "buyers_lag_1", "log_list_price", "price_vs_4w", "week_of_year", "weeks_to_christmas", "product_id"]


def christmas_gap(week: pd.Series) -> pd.Series:
    xmas = pd.to_datetime(week.dt.year.astype(str) + "-12-25")
    gap = (xmas - week).dt.days // 7
    nxt = (pd.to_datetime((week.dt.year + 1).astype(str) + "-12-25") - week).dt.days // 7
    return gap.where(gap >= 0, nxt)


def build(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["StockCode", "week"]).copy()
    g = p.groupby("StockCode")
    past = g["units"].shift(1)
    for k in LAGS:
        p[f"lag_{k}"] = g["units"].shift(k)
    pg = past.groupby(p["StockCode"])
    p["roll_mean_4"] = pg.transform(lambda s: s.rolling(4, min_periods=1).mean())
    p["roll_std_4"] = pg.transform(lambda s: s.rolling(4, min_periods=2).std())
    p["roll_mean_13"] = pg.transform(lambda s: s.rolling(13, min_periods=1).mean())
    p["level"] = pg.transform(lambda s: s.expanding().mean())
    p["zero_share_13"] = (past == 0).astype(float).where(past.notna()).groupby(p["StockCode"]).transform(
        lambda s: s.rolling(13, min_periods=1).mean())
    p["buyers_lag_1"] = g["buyers"].shift(1)
    # price is set by the retailer before the week, so the current week's list price is known in advance
    p["log_list_price"] = np.log(p["list_price"])
    p["price_vs_4w"] = p["list_price"] / g["list_price"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    p["week_of_year"] = p["week"].dt.isocalendar().week.astype(int)
    p["weeks_to_christmas"] = christmas_gap(p["week"])
    p["product_id"] = p["StockCode"].astype("category").cat.codes
    return p
