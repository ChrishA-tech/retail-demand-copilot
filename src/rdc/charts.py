"""Report charts (black and grey, readable in print)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("reports")


def main():
    res = json.loads((R / "results.json").read_text())
    bt = pd.read_csv(R / "backtest_test.csv", parse_dates=["week"])
    w = bt.groupby("week")[["units", "q100", "q500", "q900", "ma4"]].sum()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(w.index, w["q100"], w["q900"], color="0.85", label="LightGBM p10–p90 (summed)")
    ax.plot(w.index, w["units"], "k-o", ms=3, label="Actual units")
    ax.plot(w.index, w["q500"], "k--", label="LightGBM p50")
    ax.plot(w.index, w["ma4"], color="0.5", ls=":", label="4-week moving average")
    ax.set_title("Held-out test weeks: total units across 100 products"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(R / "01_backtest.png", dpi=150); plt.close(fig)

    el = res["elasticity_pooled"]
    names = {"volume_weighted_price": "Volume-weighted price\n(revenue / units)", "median_line_price": "Median line price",
             "list_price": "List price\n(lines of 12 units or fewer)"}
    keys = ["volume_weighted_price", "median_line_price", "list_price"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, k in enumerate(keys):
        e = el[k]; ax.errorbar(e["elasticity"], i, xerr=[[e["elasticity"] - e["ci95"][0]], [e["ci95"][1] - e["elasticity"]]],
                               fmt="ko", capsize=4)
        ax.text(e["ci95"][1] + 0.05, i, f'{e["elasticity"]:.2f}', va="center", fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels([names[k] for k in keys], fontsize=8); ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Price elasticity (95% CI, two-way fixed effects)")
    ax.set_title("Bulk discounts inflate elasticity when price is measured badly")
    fig.tight_layout(); fig.savefig(R / "02_elasticity_bias.png", dpi=150); plt.close(fig)

    op = res["order_policy"]; c = op["test_cost_per_week_gbp"]; champ = op["champion_chosen_on_validation"]
    order = sorted(c, key=c.get, reverse=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(order, [c[k] for k in order], color=["k" if k == champ else "0.7" for k in order])
    for i, k in enumerate(order):
        ax.text(c[k] + 80, i, f"£{c[k]:,.0f}", va="center", fontsize=9)
    ax.set_xlabel("Cost per week on test weeks (£, assumed 35% lost margin / 5% holding)")
    ax.set_title("Ordering policy chosen on validation (black) vs alternatives"); ax.set_xlim(0, max(c.values()) * 1.2)
    fig.tight_layout(); fig.savefig(R / "03_policy_cost.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
