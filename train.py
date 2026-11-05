import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import build_features

# Carrega os dados do arquivo Excel.
# read_csv é usado apenas para arquivos CSV; arquivos XLS precisam de read_excel.
df = pd.read_excel("dataset.xls", header=1, engine="xlrd")

# Remove o identificador, pois ele não ajuda a prever a inadimplência.
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Padroniza o nome da coluna que indica a inadimplência.
target_column = next(
    column for column in df.columns
    if str(column).lower().replace(".", "_").replace(" ", "_")
    in ("default_payment_next_month", "default")
)
df = df.rename(columns={target_column: "default"})

# Cria as características usadas pelo modelo
X = build_features(df)

# Define o resultado que queremos prever
y = df["default"].astype(int)

# Divide os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Cria o modelo XGBoost com configurações simples
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

# Treina o modelo
model.fit(X_train, y_train)

# Salva o modelo treinado
model.save_model("model.json")

print("Arquivo 'model.json' gerado com sucesso!")