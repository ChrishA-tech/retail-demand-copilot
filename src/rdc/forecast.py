"""Global LightGBM quantile models, weekly rolling-origin backtest and baselines."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def fit_quantile(train: pd.DataFrame, alpha: float) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **PARAMS)
    m.fit(train[FEATURES], train["units"], categorical_feature=["product_id"])
    return m


def baselines(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"naive": df["lag_1"], "ma4": df["roll_mean_4"], "seasonal_naive": df["lag_52"]},
                        index=df.index)


def backtest(feat: pd.DataFrame, origins: list[pd.Timestamp], alphas: tuple[float, ...]) -> pd.DataFrame:
    """For each forecast week, train on weeks strictly before it and predict that week (1 week ahead)."""
    usable = feat.dropna(subset=["lag_13"])  # need some history; lag_52 may be NaN (LightGBM handles it)
    out = []
    for wk in origins:
        train = usable[usable["week"] < wk]
        test = usable[usable["week"] == wk]
        if test.empty:
            continue
        res = test[["StockCode", "week", "units", "list_price"]].copy()
        for a in alphas:
            res[f"q{int(round(a * 1000)):03d}"] = np.clip(fit_quantile(train, a).predict(test[FEATURES]), 0, None)
        qcols = [f"q{int(round(a * 1000)):03d}" for a in sorted(alphas)]
        res["crossed"] = (np.diff(res[qcols].to_numpy(), axis=1) < 0).any(axis=1)
        res[qcols] = np.sort(res[qcols].to_numpy(), axis=1)  # quantile rearrangement (Chernozhukov et al., 2010)
        res = res.join(baselines(test))
        out.append(res)
    return pd.concat(out, ignore_index=True)


def wape(y, f) -> float:
    return float(np.abs(y - f).sum() / np.abs(y).sum())


def pinball(y, f, a) -> float:
    d = y - f
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def mase_scale(feat: pd.DataFrame, before: pd.Timestamp) -> float:
    h = feat[feat["week"] < before]
    return float((h["units"] - h["lag_1"]).abs().mean())
