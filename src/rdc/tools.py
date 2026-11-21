"""Tools over the serving artifact. The API and the agent both call these, so the agent can only report numbers
that the pipeline produced."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ARTIFACT = Path("artifacts/serving.json")


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    return json.loads(Path(path or ARTIFACT).read_text())


def _resolve(code_or_name: str, art: dict) -> str | None:
    """Accept a stock code or a product name (case-insensitive; a unique partial name match also works)."""
    q = str(code_or_name).strip()
    if q.upper() in art["products"]:
        return q.upper()
    ql = q.lower().strip("'\" ")
    exact = [c for c, p in art["products"].items() if p["description"].lower() == ql]
    if exact:
        return exact[0]
    part = [c for c, p in art["products"].items() if ql and ql in p["description"].lower()]
    return part[0] if len(part) == 1 else None


def _product(code: str, art: dict) -> dict | None:
    c = _resolve(code, art)
    return None if c is None else art["products"][c]


def search_products(query: str, art: dict | None = None, limit: int = 5) -> dict:
    art = art or load()
    q = query.lower().strip()
    hits = [{"stock_code": c, "description": p["description"]} for c, p in art["products"].items()
            if q in p["description"].lower() or q == c.lower()]
    return {"query": query, "matches": hits[:limit], "found": len(hits)}


def get_forecast(stock_code: str, art: dict | None = None) -> dict:
    art = art or load()
    p = _product(stock_code, art)
    if p is None:
        return {"error": f"unknown stock_code {stock_code}"}
    return {"stock_code": _resolve(stock_code, art), "description": p["description"], "week": p["week"], "units": p["forecast"],
            "last_4_weeks_units": p["last_4_weeks_units"]}


def get_elasticity(stock_code: str | None = None, art: dict | None = None) -> dict:
    art = art or load()
    pooled = art["elasticity_pooled"]
    if not stock_code:
        return {"scope": "all products (fixed effects)", "elasticity": pooled["elasticity"], "ci95": pooled["ci95"],
                "note": "association from observational prices, not a causal effect"}
    p = _product(stock_code, art)
    if p is None:
        return {"error": f"unknown stock_code {stock_code}"}
    if p["elasticity"] is None:
        return {"stock_code": stock_code, "elasticity": None, "note": "too few price changes to estimate"}
    return {"stock_code": _resolve(stock_code, art), "description": p["description"], "elasticity": p["elasticity"]["estimate"],
            "ci95": p["elasticity"]["ci95"], "note": "association from observational prices, not a causal effect"}


def get_order_recommendation(stock_code: str, art: dict | None = None) -> dict:
    art = art or load()
    p = _product(stock_code, art)
    if p is None:
        return {"error": f"unknown stock_code {stock_code}"}
    return {"stock_code": _resolve(stock_code, art), "description": p["description"], "week": p["week"],
            "policy": p["recommended_order"]["policy"], "order_units": p["recommended_order"]["units"],
            "list_price_gbp": p["list_price"]}


def get_policy_costs(art: dict | None = None) -> dict:
    art = art or load()
    return {"backtest_cost_per_week_gbp": art["test_cost_per_week_gbp"], "champion": art["champion_policy"],
            "note": "12 held-out weeks; cost rates are assumptions (35% lost margin, 5% holding)"}


REGISTRY = {
    "search_products": search_products,
    "get_forecast": get_forecast,
    "get_elasticity": get_elasticity,
    "get_order_recommendation": get_order_recommendation,
    "get_policy_costs": get_policy_costs,
}

SCHEMAS = [
    {"type": "function", "function": {"name": "search_products", "description": "Find products by name or stock code.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_forecast", "description": "Next-week demand forecast (p10, p50, p90 units) for one product, by stock code or product name.",
     "parameters": {"type": "object", "properties": {"stock_code": {"type": "string"}}, "required": ["stock_code"]}}},
    {"type": "function", "function": {"name": "get_elasticity", "description": "Price elasticity for one product, or for all products if stock_code is omitted.",
     "parameters": {"type": "object", "properties": {"stock_code": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_order_recommendation", "description": "Recommended order quantity for next week for one product, by stock code or product name.",
     "parameters": {"type": "object", "properties": {"stock_code": {"type": "string"}}, "required": ["stock_code"]}}},
    {"type": "function", "function": {"name": "get_policy_costs", "description": "Backtest cost per week of each ordering policy.",
     "parameters": {"type": "object", "properties": {}}}},
]
