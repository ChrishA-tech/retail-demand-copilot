import numpy as np
import pandas as pd

from rdc import agent, data, features, inventory


def _panel():
    weeks = pd.date_range("2010-01-04", periods=60, freq="W-MON")
    rows = []
    for code, base in [("A1", 10), ("B2", 50)]:
        for i, w in enumerate(weeks):
            rows.append({"StockCode": code, "week": w, "units": base + i, "revenue": (base + i) * 2.0,
                         "median_price": 2.0, "list_price": 2.0 + (i % 5) * 0.1, "buyers": 3, "lines": 4,
                         "description": code})
    return pd.DataFrame(rows)


def test_features_never_see_current_week():
    f = features.build(_panel())
    a = f[f["StockCode"] == "A1"].reset_index(drop=True)
    assert a.loc[10, "lag_1"] == a.loc[9, "units"]
    assert a.loc[10, "roll_mean_4"] == a.loc[6:9, "units"].mean()
    assert a.loc[10, "level"] == a.loc[0:9, "units"].mean()
    # changing this week's units must not change any of this week's features
    p2 = _panel(); p2.loc[(p2["StockCode"] == "A1") & (p2.index == 10), "units"] = 10_000
    f2 = features.build(p2)
    cols = [c for c in features.FEATURES if c != "product_id"]
    pd.testing.assert_frame_equal(f.loc[10, cols].to_frame(), f2.loc[10, cols].to_frame())


def test_clean_removes_cancellations_and_duplicates():
    df = pd.DataFrame({"Invoice": ["1", "C2", "3", "3"], "StockCode": ["X", "X", "POST", "Y"],
                       "Quantity": [1, -1, 1, 2], "Price": [1.0, 1.0, 5.0, 1.0],
                       "InvoiceDate": pd.to_datetime(["2010-01-01"] * 4), "CustomerID": [1, 1, 1, 1],
                       "Description": ["a"] * 4, "Country": ["UK"] * 4})
    out, log = data.clean(pd.concat([df, df.iloc[[3]]]))
    assert log["cancelled_invoices"] == 1 and log["non_product_codes"] == 1 and log["exact_duplicates"] == 1
    assert len(out) == 2


def test_newsvendor_cost_and_ratio():
    assert abs(inventory.CRITICAL_RATIO - 0.875) < 1e-9
    c = inventory.cost(np.array([10, 10]), np.array([8, 12]), np.array([1.0, 1.0]))
    assert np.allclose(c, [0.70, 0.10])


def test_verifier_flags_invented_numbers():
    ev = [{"units": {"p50": 120.4}}]
    assert agent.unsupported("Expect about 120 units.", ev, "q") == []
    assert agent.unsupported("Expect 999 units.", ev, "q") == [999.0]
    assert agent.unsupported("Critical ratio is 87.5%.", [{"r": 0.875}], "q") == []


def test_verifier_allows_confidence_level():
    assert agent.unsupported("Elasticity is -0.80 (95% confidence interval -1.59 to -0.56).",
                             [{"e": -0.796, "ci95": [-1.589, -0.558]}], "q") == []


def test_tools_resolve_product_names():
    from rdc import tools
    art = {"products": {"82482": {"description": "WOODEN PICTURE FRAME WHITE FINISH", "week": "w",
                                  "forecast": {"p10": 1, "p50": 2, "p90": 3}, "last_4_weeks_units": []}}}
    assert tools.get_forecast("Wooden Picture Frame White Finish", art)["stock_code"] == "82482"
    assert "error" in tools.get_forecast("No Such Thing", art)
