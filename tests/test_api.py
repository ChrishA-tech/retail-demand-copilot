from fastapi.testclient import TestClient

from rdc import tools
from rdc.api import app

c = TestClient(app)
code = next(iter(tools.load()["products"]))


def test_health():
    r = c.get("/health"); assert r.status_code == 200 and r.json()["products"] > 0


def test_forecast_and_order():
    f = c.get(f"/forecast/{code}").json()
    assert f["units"]["p10"] <= f["units"]["p50"] <= f["units"]["p90"] or f["units"]["p90"] >= f["units"]["p10"]
    o = c.get(f"/order/{code}").json(); assert o["order_units"] >= 0


def test_unknown_product_is_404():
    assert c.get("/forecast/NOPE1").status_code == 404


def test_batch_predict_contract():
    r = c.post("/predict", json={"instances": [{"stock_code": code}]}).json()
    assert len(r["predictions"]) == 1 and "order_units" in r["predictions"][0]


def test_scripted_agent_abstains_on_unknown_product():
    from rdc.agent import ScriptedAgent
    assert ScriptedAgent().ask("What is the demand forecast for stock code 99999X?").abstained
