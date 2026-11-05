"""Features compartilhadas por train.py e app.py (evita divergência entre treino e produção)."""
import pandas as pd

# Mês 1 = mais recente (setembro/2005 no dataset original) ... mês 6 = mais antigo (abril/2005)
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]        # situação do pagamento
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]                        # valor da fatura
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]                         # valor pago


def build_features(df):
    X = df[["LIMIT_BAL", "AGE"] + PAY + BILL + PAID].astype(float).copy()
    X["utilization"] = df["BILL_AMT1"] / df["LIMIT_BAL"]                       # uso do limite
    X["avg_status"] = df[PAY].mean(axis=1)                                     # atraso médio
    X["max_status"] = df[PAY].max(axis=1)                                      # pior atraso
    X["n_late"] = (df[PAY] > 0).sum(axis=1)                                    # meses em atraso
    X["paid_ratio"] = (df[PAID].sum(axis=1) / df[BILL].sum(axis=1).clip(lower=1)).clip(0, 2)
    return X


def frame_from_inputs(limit_bal, age, pay_status, bill_amt, pay_amt):
    """Monta 1 linha no formato do dataset a partir das listas de 6 meses (mais recente primeiro)."""
    row = {"LIMIT_BAL": limit_bal, "AGE": age}
    row.update(zip(PAY, pay_status))
    row.update(zip(BILL, bill_amt))
    row.update(zip(PAID, pay_amt))
    return pd.DataFrame([row])
