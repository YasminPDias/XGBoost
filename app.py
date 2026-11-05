import json
import os

import xgboost as xgb
from flask import Flask, jsonify, request
from flask_cors import CORS
from xgboost import XGBClassifier

from features import BILL, PAID, PAY, build_features, frame_from_inputs

app = Flask(__name__)
CORS(app)  # produção: CORS(app, origins=["https://seu-front.vercel.app"])

model = XGBClassifier()
model.load_model("model.json")
CFG = json.load(open("config.json"))

LABELS = {"LIMIT_BAL": "Limite de crédito", "AGE": "Idade", "utilization": "Uso do limite",
          "avg_status": "Atraso médio", "max_status": "Pior atraso", "n_late": "Meses em atraso",
          "paid_ratio": "Proporção da fatura paga"}
for i in range(6):
    LABELS[PAY[i]] = f"Situação do pagamento (mês {i + 1})"
    LABELS[BILL[i]] = f"Fatura (mês {i + 1})"
    LABELS[PAID[i]] = f"Valor pago (mês {i + 1})"


def parse(d):
    limit, age = float(d["limit_bal"]), float(d["age"])
    arr = {k: [float(v) for v in d[k]] for k in ("pay_status", "bill_amt", "pay_amt")}
    if any(len(v) != 6 for v in arr.values()):
        raise ValueError("Informe 6 meses em pay_status, bill_amt e pay_amt (mais recente primeiro).")
    if limit <= 0 or not 18 <= age <= 100:
        raise ValueError("limit_bal deve ser > 0 e age entre 18 e 100.")
    if any(not -2 <= s <= 9 for s in arr["pay_status"]) or any(a < 0 for a in arr["pay_amt"]):
        raise ValueError("pay_status entre -2 e 9; pay_amt não pode ser negativo.")
    return limit, age, arr


@app.route("/")
def home():
    return "API ITS WORKS!"


@app.route("/api/model-info")
def info():
    return jsonify(json.load(open("metrics.json")))


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        limit, age, arr = parse(request.get_json(silent=True) or {})
    except KeyError as e:
        return jsonify({"erro": f"Campo ausente: {e.args[0]}"}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"erro": str(e)}), 400

    X = build_features(frame_from_inputs(limit, age, arr["pay_status"], arr["bill_amt"], arr["pay_amt"]))[CFG["features"]]
    p = float(model.predict_proba(X)[0, 1])
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    factors = sorted(({"feature": k, "label": LABELS.get(k, k), "value": round(float(X[k].iloc[0]), 3),
                       "contribution": round(float(c), 4)} for k, c in zip(CFG["features"], contrib)),
                     key=lambda f: -abs(f["contribution"]))[:6]
    return jsonify({"probability_default": round(p, 4), "flagged": p >= CFG["threshold"],
                    "threshold": round(CFG["threshold"], 4), "base_rate": round(CFG["base_rate"], 4),
                    "factors": factors})  # contribution > 0 aumenta o risco


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
