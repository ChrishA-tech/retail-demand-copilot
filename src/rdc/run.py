"""End-to-end run: clean -> panel -> elasticity -> backtest -> order policies -> serving artifact -> report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, elasticity, features, forecast, inventory

ALPHAS = (0.1, 0.5, inventory.CRITICAL_RATIO, 0.9)
QCOL = {a: f"q{int(round(a * 1000)):03d}" for a in ALPHAS}


def accuracy(bt: pd.DataFrame, scale: float) -> dict:
    y = bt["units"].to_numpy()
    out = {}
    for name, col in [("lightgbm_median", QCOL[0.5]), ("moving_average_4w", "ma4"), ("naive", "naive"),
                      ("seasonal_naive", "seasonal_naive")]:
        f = bt[col].fillna(bt["ma4"]).to_numpy()
        out[name] = {"wape": round(forecast.wape(y, f), 4), "mase": round(float(np.abs(y - f).mean() / scale), 3),
                     "bias": round(float((f - y).sum() / y.sum()), 4)}
    lo, hi = bt[QCOL[0.1]].to_numpy(), bt[QCOL[0.9]].to_numpy()
    out["interval_p10_p90_coverage"] = round(float(np.mean((y >= lo) & (y <= hi))), 4)
    out["pinball"] = {QCOL[a]: round(forecast.pinball(y, bt[QCOL[a]].to_numpy(), a), 2) for a in ALPHAS}
    return out


def product_bootstrap_wape(bt: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g = bt.assign(ea=(bt["units"] - bt[a]).abs(), eb=(bt["units"] - bt[b]).abs()).groupby("StockCode")[
        ["ea", "eb", "units"]].sum()
    diffs = []
    for _ in range(n):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        diffs.append((s["ea"].sum() - s["eb"].sum()) / s["units"].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = (g["ea"].sum() - g["eb"].sum()) / g["units"].sum()
    return {"wape_diff_pp": round(100 * point, 2), "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)]}


def policies(bt: pd.DataFrame, k_ma: float, k_med: float) -> pd.DataFrame:
    u, p = bt["units"].to_numpy(), bt["list_price"].to_numpy()
    orders = {"ma4_x1": bt["ma4"], f"ma4_x{k_ma}": k_ma * bt["ma4"], "lgbm_median": bt[QCOL[0.5]],
              f"lgbm_median_x{k_med}": k_med * bt[QCOL[0.5]], "lgbm_q875_newsvendor": bt[QCOL[inventory.CRITICAL_RATIO]]}
    out = bt[["StockCode", "week", "units", "list_price"]].copy()
    for k, o in orders.items():
        out[k] = inventory.cost(u, o.to_numpy(), p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/onlineretail2.rda")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--quick", action="store_true", help="small run for CI: 30 products, 4+4 weeks")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    art = Path("artifacts"); art.mkdir(exist_ok=True)

    raw = data.load_raw(a.data)
    clean, log = data.clean(raw)
    panel = data.weekly_panel(clean, 30 if a.quick else a.top)
    feat = features.build(panel)
    weeks = sorted(feat["week"].unique())
    n_test = 4 if a.quick else 12
    test_w, val_w = weeks[-n_test:], weeks[-2 * n_test:-n_test]

    el_pooled = elasticity.pooled(panel)
    el_prod = elasticity.per_product(panel, n_boot=100 if a.quick else 500)
    el_prod.to_csv(out / "elasticity_by_product.csv", index=False)

    bt_val = forecast.backtest(feat, val_w, ALPHAS)
    bt_test = forecast.backtest(feat, test_w, ALPHAS)
    bt_test.to_csv(out / "backtest_test.csv", index=False)
    scale = forecast.mase_scale(feat, val_w[0])

    k_ma = inventory.tune_multiplier(bt_val, "ma4")
    k_med = inventory.tune_multiplier(bt_val, QCOL[0.5])
    pol = policies(bt_test, k_ma, k_med)
    pol_cols = [c for c in pol.columns if c not in {"StockCode", "week", "units", "list_price"}]
    cost_week = {c: round(float(pol[c].sum() / pol["week"].nunique()), 1) for c in pol_cols}
    best_baseline = f"ma4_x{k_ma}"

    results = {
        "data": {"source": "UCI Online Retail II (Chen, 2012), CC BY 4.0", **log, "products": int(panel["StockCode"].nunique()),
                 "weeks": len(weeks), "first_week": str(weeks[0].date()), "last_week": str(weeks[-1].date()),
                 "zero_weeks_share": round(float((panel["units"] == 0).mean()), 4)},
        "protocol": {"validation_weeks": [str(val_w[0].date()), str(val_w[-1].date())],
                     "test_weeks": [str(test_w[0].date()), str(test_w[-1].date())],
                     "retrain": "every week, on data strictly before the forecast week", "horizon_weeks": 1},
        "elasticity_pooled": el_pooled,
        "elasticity_by_product": {"estimable": int(el_prod["estimable"].sum()), "products": len(el_prod),
                                  "ci_entirely_below_zero": int((el_prod["ci_high"] < 0).sum()),
                                  "median": float(el_prod["elasticity"].median())},
        "accuracy_validation": accuracy(bt_val, scale),
        "accuracy_test": accuracy(bt_test, scale),
        "quantile_crossing_rate_before_rearrangement": round(float(bt_test["crossed"].mean()), 4),
        "wape_lgbm_minus_ma4_test": product_bootstrap_wape(bt_test, QCOL[0.5], "ma4"),
        "order_policy": {"cu": inventory.CU, "co": inventory.CO, "critical_ratio": inventory.CRITICAL_RATIO,
                         "multipliers_tuned_on_validation": {"ma4": k_ma, "lgbm_median": k_med},
                         "test_cost_per_week_gbp": cost_week,
                         "newsvendor_vs_best_baseline": inventory.block_bootstrap_diff(pol, "lgbm_q875_newsvendor", best_baseline),
                         "newsvendor_vs_ma4_x1": inventory.block_bootstrap_diff(pol, "lgbm_q875_newsvendor", "ma4_x1")},
    }
    pol_val = policies(bt_val, k_ma, k_med)
    val_cost = {c: round(float(pol_val[c].sum() / pol_val["week"].nunique()), 1) for c in pol_cols}
    champion = min(val_cost, key=val_cost.get)  # chosen on validation weeks, never on the test weeks
    results["order_policy"]["validation_cost_per_week_gbp"] = val_cost
    results["order_policy"]["champion_chosen_on_validation"] = champion
    results["order_policy"]["champion_vs_ma4_x1_test"] = inventory.block_bootstrap_diff(pol, champion, "ma4_x1")
    results["order_policy"]["champion_vs_ma4_safety_stock_test"] = inventory.block_bootstrap_diff(pol, champion, best_baseline)

    # serving artifact: train on all weeks, forecast the week after the last one
    nxt = weeks[-1] + pd.Timedelta(weeks=1)
    last = panel[panel["week"] == weeks[-1]].copy()
    last["week"], last["units"] = nxt, np.nan
    feat_all = features.build(pd.concat([panel, last], ignore_index=True))
    train, fut = feat_all[feat_all["week"] < nxt].dropna(subset=["lag_13", "units"]), feat_all[feat_all["week"] == nxt]
    srt = sorted(ALPHAS)
    mat = np.sort(np.column_stack([np.clip(forecast.fit_quantile(train, a_).predict(fut[features.FEATURES]), 0, None)
                                   for a_ in srt]), axis=1)
    preds = {a_: mat[:, i] for i, a_ in enumerate(srt)}
    el_map = el_prod.set_index("StockCode")
    products = {}
    for i, (_, r) in enumerate(fut.iterrows()):
        e = el_map.loc[r["StockCode"]]
        q = {QCOL[a_]: round(float(preds[a_][i]), 1) for a_ in ALPHAS}
        rec = {"lgbm_q875_newsvendor": q[QCOL[inventory.CRITICAL_RATIO]], f"ma4_x{k_ma}": round(k_ma * float(r["roll_mean_4"]), 1),
               "ma4_x1": round(float(r["roll_mean_4"]), 1), "lgbm_median": q[QCOL[0.5]],
               f"lgbm_median_x{k_med}": round(k_med * q[QCOL[0.5]], 1)}
        products[r["StockCode"]] = {
            "description": r["description"], "week": str(nxt.date()), "list_price": round(float(r["list_price"]), 2),
            "forecast": {"p10": q[QCOL[0.1]], "p50": q[QCOL[0.5]], "p90": q[QCOL[0.9]]},
            "last_4_weeks_units": [float(x) for x in panel[panel["StockCode"] == r["StockCode"]]["units"].tail(4)],
            "elasticity": None if not e["estimable"] else {"estimate": float(e["elasticity"]), "ci95": [float(e["ci_low"]), float(e["ci_high"])]},
            "recommended_order": {"policy": champion, "units": rec[champion]},
        }
    serving = {"generated_for_week": str(nxt.date()), "champion_policy": champion, "products": products,
               "elasticity_pooled": el_pooled["list_price"], "test_cost_per_week_gbp": cost_week}
    (art / "serving.json").write_text(json.dumps(serving, indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    if os.environ.get("RDC_MLFLOW", "1") == "1" and not a.quick:
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
            mlflow.set_experiment("retail-demand-copilot")
            with mlflow.start_run(run_name="backtest"):
                mlflow.log_params({**forecast.PARAMS, "top_products": a.top, "test_weeks": n_test})
                for k, v in results["accuracy_test"].items():
                    if isinstance(v, dict) and "wape" in v:
                        mlflow.log_metric(f"test_wape_{k}", v["wape"])
                for k, v in cost_week.items():
                    mlflow.log_metric(f"test_cost_{k}", v)
                mlflow.log_artifact(str(out / "results.json"))
        except Exception as ex:  # tracking is optional
            print("mlflow skipped:", ex)
    print(json.dumps({k: results[k] for k in ["accuracy_test", "wape_lgbm_minus_ma4_test", "order_policy"]}, indent=1))
    return results


if __name__ == "__main__":
    main()
