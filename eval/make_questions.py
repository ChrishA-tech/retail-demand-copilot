"""Build the agent evaluation set from the serving artifact. Every answerable question has a known numeric answer."""
import json
import random
from pathlib import Path

art = json.loads(Path("artifacts/serving.json").read_text())
rng = random.Random(7)
codes = sorted(art["products"])
rng.shuffle(codes)
qs = []
for c in codes[:8]:
    p = art["products"][c]
    qs.append({"id": f"f50_{c}", "type": "forecast_p50", "question": f"What is the median (p50) demand forecast for next week for stock code {c}?", "expected": [p["forecast"]["p50"]]})
for c in codes[8:14]:
    p = art["products"][c]
    qs.append({"id": f"f90_{c}", "type": "forecast_p90", "question": f"How many units of '{p['description'].title()}' could we sell next week in a high case (p90)?", "expected": [p["forecast"]["p90"]]})
for c in codes[14:22]:
    p = art["products"][c]
    qs.append({"id": f"ord_{c}", "type": "order", "question": f"How many units of {c} should we order for next week?", "expected": [p["recommended_order"]["units"]]})
el = [c for c in codes[22:] if art["products"][c]["elasticity"]][:6]
for c in el:
    qs.append({"id": f"el_{c}", "type": "elasticity", "question": f"What is the price elasticity of {c}?", "expected": [art["products"][c]["elasticity"]["estimate"]]})
pe = art["elasticity_pooled"]["elasticity"]
qs += [{"id": "el_pooled_1", "type": "elasticity_pooled", "question": "Across all products, how sensitive is demand to price (elasticity)?", "expected": [pe]},
       {"id": "el_pooled_2", "type": "elasticity_pooled", "question": "Give me the overall price elasticity estimate.", "expected": [pe]}]
champ = art["champion_policy"]; cw = art["test_cost_per_week_gbp"][champ]
qs += [{"id": "pol_1", "type": "policy", "question": "Which ordering policy was cheapest in the backtest, and what did it cost per week?", "expected": [cw]},
       {"id": "pol_2", "type": "policy", "question": "What is the weekly cost of the best ordering policy?", "expected": [cw]}]
no_el = [c for c in codes if art["products"][c]["elasticity"] is None][:2]
unans = ["What will revenue be next year (2013) for stock code {c}?".format(c=codes[0]),
         "What is the demand forecast for stock code 99999X?",
         "How many units of 00000 should we order?",
         "If a competitor cuts prices by 10%, how many units of {c} will we lose?".format(c=codes[1]),
         "Will the weather next week change demand for {c}?".format(c=codes[2]),
         "What caused the drop in sales of {c} last spring?".format(c=codes[3])]
unans += [f"What is the price elasticity of {c}?" for c in no_el]
qs += [{"id": f"abstain_{i}", "type": "abstain", "question": q, "expected": None} for i, q in enumerate(unans)]
Path("eval/questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
print(len(qs), "questions,", sum(q["expected"] is None for q in qs), "unanswerable")
