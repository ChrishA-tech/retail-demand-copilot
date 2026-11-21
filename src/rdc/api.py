"""FastAPI tool service. Run: uvicorn rdc.api:app --port 8080"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import tools

app = FastAPI(title="Retail Demand Copilot tools", version="1.0.0")


def _ok(res: dict) -> dict:
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health() -> dict:
    art = tools.load()
    return {"status": "ok", "products": len(art["products"]), "week": art["generated_for_week"]}


@app.get("/products")
def products(query: str = "") -> dict:
    return tools.search_products(query)


@app.get("/forecast/{stock_code}")
def forecast(stock_code: str) -> dict:
    return _ok(tools.get_forecast(stock_code))


@app.get("/elasticity")
def elasticity_all() -> dict:
    return tools.get_elasticity(None)


@app.get("/elasticity/{stock_code}")
def elasticity(stock_code: str) -> dict:
    return _ok(tools.get_elasticity(stock_code))


@app.get("/order/{stock_code}")
def order(stock_code: str) -> dict:
    return _ok(tools.get_order_recommendation(stock_code))


@app.get("/policies")
def policies() -> dict:
    return tools.get_policy_costs()


class Batch(BaseModel):
    instances: list[dict]


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
def predict(batch: Batch) -> dict:
    """Vertex AI / SageMaker style batch route: {"instances": [{"stock_code": ...}]} -> {"predictions": [...]}"""
    return {"predictions": [tools.get_order_recommendation(i.get("stock_code", "")) | tools.get_forecast(i.get("stock_code", ""))
                            for i in batch.instances]}
