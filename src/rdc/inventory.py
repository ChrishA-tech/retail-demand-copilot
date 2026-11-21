"""Turn forecasts into order quantities and score them in pounds.

Newsvendor cost per product-week: under-stock loses margin cu = 35% of list price per unit short; over-stock costs
co = 5% of list price per unit left (holding and markdown). These rates are assumptions, not the retailer's numbers.
Optimal order = the cu / (cu + co) quantile of demand = 0.875.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CU, CO = 0.35, 0.05
CRITICAL_RATIO = CU / (CU + CO)


def cost(units: np.ndarray, order: np.ndarray, price: np.ndarray) -> np.ndarray:
    short = np.maximum(units - order, 0)
    left = np.maximum(order - units, 0)
    return CU * price * short + CO * price * left


def tune_multiplier(df: pd.DataFrame, col: str, grid=np.round(np.arange(0.3, 3.01, 0.05), 2)) -> float:
    """Pick the safety multiplier k on order = k * forecast that minimises cost on the validation weeks."""
    costs = [cost(df["units"].to_numpy(), k * df[col].to_numpy(), df["list_price"].to_numpy()).sum() for k in grid]
    return float(grid[int(np.argmin(costs))])


def block_bootstrap_diff(df: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0) -> dict:
    """95% CI for the weekly cost difference (a - b), resampling whole products to respect within-product correlation."""
    rng = np.random.default_rng(seed)
    per = df.groupby("StockCode")[[a, b]].sum()
    weeks = df["week"].nunique()
    diffs = per[a].to_numpy() - per[b].to_numpy()
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].sum() / weeks for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff_per_week": round(diffs.sum() / weeks, 1), "ci95": [round(lo, 1), round(hi, 1)],
            "share_of_boots_below_zero": round(float(np.mean(np.array(boots) < 0)), 3)}
