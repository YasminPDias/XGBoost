"""Inadimplência de cartão de crédito (UCI #350). Uso: python train.py
Lê dataset.csv se existir; senão baixa do UCI via ucimlrepo."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance

from features import BILL, PAID, PAY, build_features

SEED = 42
RAW = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY + BILL + PAID
XGB_GRID = {"max_depth": [2, 3, 4, 5, 6], "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.02, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10],
            "reg_lambda": [1, 3, 10]}


def load():
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
    else:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=350)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    df = df.drop(columns=[c for c in df.columns if str(c).upper() == "ID"])
    if "X1" in df.columns:  # nomes genéricos do UCI -> nomes descritivos
        df = df.rename(columns=dict(zip([f"X{i}" for i in range(1, 24)], RAW)))
    df = df.rename(columns={"PAY_1": "PAY_0"})
    tgt = next(c for c in df.columns if str(c).lower().replace(".", "_").replace(" ", "_")
               in ("default_payment_next_month", "y", "default"))
    return df.rename(columns={tgt: "default"}).dropna()


def tune(est, grid, X, y, cv, n_iter=None):
    kw = dict(cv=cv, scoring="roc_auc", n_jobs=-1)
    s = (RandomizedSearchCV(est, grid, n_iter=n_iter, random_state=SEED, **kw) if n_iter
         else GridSearchCV(est, grid, **kw))
    return s.fit(X, y)


def best_threshold(y, p):
    """Limiar que maximiza o F1 (escolhido só com previsões out-of-fold do treino)."""
    pr, rc, th = precision_recall_curve(y, p)
    return float(th[(2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)).argmax()])


def bootstrap_ci(y, p_a, p_b, n=500):
    """IC 95% do AUC de A e da diferença AUC(A) - AUC(B), reamostrando o teste."""
    rng, y, aucs, diffs = np.random.default_rng(SEED), np.asarray(y), [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a, b = roc_auc_score(y[i], p_a[i]), roc_auc_score(y[i], p_b[i])
        aucs.append(a); diffs.append(a - b)
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(diffs, [2.5, 97.5])


def main():
    df = load()
    y, X = df["default"].astype(int), build_features(df)
    print(f"{len(df)} clientes | inadimplentes: {y.mean():.1%}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    models = {
        "Regressão logística": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                {"logisticregression__C": [0.1, 1, 10]}, None),
        "Random Forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                 random_state=SEED, n_jobs=1),
                          {"max_depth": [6, 10, None]}, None),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1), XGB_GRID, 20),
    }
    rows, probas, best = [], {}, {}
    # Baseline de domínio: "atrasou no último mês => vai inadimplir"
    pb = (Xte["PAY_0"] > 0).astype(int)
    rows.append({"modelo": "Regra: atraso no último mês", "accuracy": accuracy_score(yte, pb),
                 "precision": precision_score(yte, pb), "recall": recall_score(yte, pb),
                 "f1": f1_score(yte, pb)})
    rows.append({"modelo": "Sempre 'não inadimplente'", "accuracy": 1 - yte.mean()})

    for name, (est, grid, n_iter) in models.items():
        print(f"\nAjustando {name}...")
        s = tune(est, grid, Xtr, ytr, cv, n_iter)
        m = s.best_estimator_
        oof = cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr, p = best_threshold(ytr, oof), m.predict_proba(Xte)[:, 1]
        pred = (p >= thr).astype(int)
        rows.append({"modelo": name, "auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                     "logloss": log_loss(yte, p), "accuracy": accuracy_score(yte, pred),
                     "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
                     "f1": f1_score(yte, pred), "limiar": thr, "cv_auc": s.best_score_})
        probas[name], best[name] = p, (m, thr, s.best_params_)
        print(f"  CV AUC={s.best_score_:.4f} | params={s.best_params_}")

    table = pd.DataFrame(rows).round(3)
    print("\n=== TESTE ===\n" + table.to_string(index=False, na_rep="-"))

    xgb, thr, params = best["XGBoost"]
    auc_ci, diff_ci = bootstrap_ci(yte, probas["XGBoost"], probas["Regressão logística"])
    print(f"\nAUC XGBoost IC95%: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print(f"AUC(XGB) - AUC(Reg. logística) IC95%: [{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]"
          "  (se incluir 0, a diferença pode ser ruído)")
    cm = confusion_matrix(yte, (probas["XGBoost"] >= thr).astype(int))
    print("Matriz de confusão (XGBoost):\n", cm)

    plot_importance(xgb, importance_type="gain", max_num_features=15, title="Importância (gain)")
    plt.savefig("feature_importance.png", bbox_inches="tight", dpi=150)
    xgb.save_model("model.json")
    json.dump({"features": list(X.columns), "threshold": thr, "base_rate": float(y.mean())},
              open("config.json", "w"))
    json.dump({"table": table.to_dict("records"), "best_params": params, "confusion_matrix": cm.tolist(),
               "auc_ci95": auc_ci.tolist(), "auc_diff_vs_logreg_ci95": diff_ci.tolist()},
              open("metrics.json", "w"), indent=2)
    print("\nGerados: model.json, config.json, metrics.json, feature_importance.png")


if __name__ == "__main__":
    main()
