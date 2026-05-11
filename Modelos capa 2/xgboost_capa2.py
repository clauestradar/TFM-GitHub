# ============================================================
# XGBOOST - CAPA 2 TFM
# Split temporal por meses completos
# Train / Test / Validation / Backtesting
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# 1. CARGAR DATASET
# ============================================================

df = pd.read_csv("analytic_panel_final_FINAL.csv")

# Convertir fecha
df["month"] = pd.to_datetime(df["month"])

# Ordenar cronológicamente
df = df.sort_values(["month", "asin"]).reset_index(drop=True)

print("Shape dataset completo:", df.shape)
print("Fecha mínima:", df["month"].min())
print("Fecha máxima:", df["month"].max())

# ============================================================
# 2. OBTENER MESES ÚNICOS ORDENADOS
# ============================================================

months = np.array(sorted(df["month"].unique()))
n_months = len(months)

print("\nNúmero total de meses:", n_months)

# ============================================================
# 3. SEPARAR 10% FINAL PARA BACKTESTING
# ============================================================

n_backtest_months = int(n_months * 0.10)

backtest_months = months[-n_backtest_months:]
modeling_months = months[:-n_backtest_months]

# ============================================================
# 4. DEL 90%, SEPARAR 30% FINAL PARA VALIDACIÓN
# ============================================================

n_modeling_months = len(modeling_months)
n_validation_months = int(n_modeling_months * 0.30)

validation_months = modeling_months[-n_validation_months:]
train_test_months = modeling_months[:-n_validation_months]

# ============================================================
# 5. DEL BLOQUE RESTANTE, SEPARAR 70% TRAIN Y 30% TEST
# ============================================================

n_train_test_months = len(train_test_months)
n_test_months = int(n_train_test_months * 0.30)

test_months = train_test_months[-n_test_months:]
train_months = train_test_months[:-n_test_months]

# ============================================================
# 6. CREAR DATAFRAMES FINALES
# ============================================================

df_train = df[df["month"].isin(train_months)].copy()
df_test = df[df["month"].isin(test_months)].copy()
df_validation = df[df["month"].isin(validation_months)].copy()
df_backtest = df[df["month"].isin(backtest_months)].copy()

# ============================================================
# 7. REVISAR RESULTADOS DEL SPLIT
# ============================================================

print("\n--- SPLIT TEMPORAL POR MESES COMPLETOS ---")

print("\nTrain:")
print("Shape:", df_train.shape)
print("Fechas:", df_train["month"].min(), "→", df_train["month"].max())
print("Meses:", df_train["month"].nunique())

print("\nTest:")
print("Shape:", df_test.shape)
print("Fechas:", df_test["month"].min(), "→", df_test["month"].max())
print("Meses:", df_test["month"].nunique())

print("\nValidation:")
print("Shape:", df_validation.shape)
print("Fechas:", df_validation["month"].min(), "→", df_validation["month"].max())
print("Meses:", df_validation["month"].nunique())

print("\nBacktesting:")
print("Shape:", df_backtest.shape)
print("Fechas:", df_backtest["month"].min(), "→", df_backtest["month"].max())
print("Meses:", df_backtest["month"].nunique())

# ============================================================
# 8. COMPROBACIÓN FINAL
# ============================================================

total_split = (
    len(df_train)
    + len(df_test)
    + len(df_validation)
    + len(df_backtest)
)

print("\n--- COMPROBACIÓN FINAL ---")
print("Total original:", len(df))
print("Total dividido:", total_split)
print("¿Coinciden?:", len(df) == total_split)

print("\nPorcentaje aproximado sobre dataset original:")
print("Train:", round(len(df_train) / len(df) * 100, 2), "%")
print("Test:", round(len(df_test) / len(df) * 100, 2), "%")
print("Validation:", round(len(df_validation) / len(df) * 100, 2), "%")
print("Backtesting:", round(len(df_backtest) / len(df) * 100, 2), "%")
# ============================================================
# 9. DEFINIR TARGET
# ============================================================

target = "log_price"

# ============================================================
# 10. COLUMNAS A EXCLUIR
# ============================================================

columns_to_drop = [
    "log_price",          # target
    "target_price",       # versión original del target
    "month",              # fecha
    "asin",               # identificador único
    "product_title"       # texto libre
]
# ============================================================
# 11. CREAR X E Y
# ============================================================

print("\nColumnas disponibles en el dataset:")
print(df_train.columns.tolist())

target = "price_log"

columns_to_drop = [
    target,
    "month",
    "asin",

    # precios directos
    "amazon_price",
    "amazon_price_raw",
    "amazon_price_final",
    "list_price",
    "list_price_raw",
    "list_price_final",

    # leakage por ingeniería
    "price_log",
    "price_change_pct",
    "discount_rate",
    "category_mean_price",
    "price_vs_category_mean",
    "real_price",
    "demand_proxy",

    # leakage por ranking/demanda
    "sales_rank_raw",
    "sales_rank_log",
    "sales_rank_change",

    # metadata de imputación
    "amazon_price_imputed"
]

columns_to_drop_existing = [
    col for col in columns_to_drop if col in df_train.columns
]

print("\nColumnas eliminadas:")
print(columns_to_drop_existing)

X_train = df_train.drop(columns=columns_to_drop_existing)
y_train = df_train[target]

X_test = df_test.drop(columns=columns_to_drop_existing)
y_test = df_test[target]

X_validation = df_validation.drop(columns=columns_to_drop_existing)
y_validation = df_validation[target]

X_backtest = df_backtest.drop(columns=columns_to_drop_existing)
y_backtest = df_backtest[target]

# ============================================================
# ELIMINAR FILAS CON TARGET NULO
# ============================================================

train_mask = y_train.notnull()
test_mask = y_test.notnull()
validation_mask = y_validation.notnull()
backtest_mask = y_backtest.notnull()

X_train = X_train[train_mask].copy()
y_train = y_train[train_mask].copy()

X_test = X_test[test_mask].copy()
y_test = y_test[test_mask].copy()

X_validation = X_validation[validation_mask].copy()
y_validation = y_validation[validation_mask].copy()

X_backtest = X_backtest[backtest_mask].copy()
y_backtest = y_backtest[backtest_mask].copy()

print("\nDespués de eliminar targets nulos:")
print("Train:", X_train.shape, "| y nulos:", y_train.isnull().sum())
print("Test:", X_test.shape, "| y nulos:", y_test.isnull().sum())
print("Validation:", X_validation.shape, "| y nulos:", y_validation.isnull().sum())
print("Backtest:", X_backtest.shape, "| y nulos:", y_backtest.isnull().sum())

# ============================================================
# 12. DETECTAR VARIABLES CATEGÓRICAS
# ============================================================

categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nVariables categóricas:")
print(categorical_cols)

# ============================================================
# 13. LABEL ENCODING
# ============================================================

from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    # juntar todos los posibles valores
    combined_values = pd.concat([
        X_train[col],
        X_test[col],
        X_validation[col],
        X_backtest[col]
    ]).astype(str)

    le.fit(combined_values)

    # transformar
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    X_validation[col] = le.transform(X_validation[col].astype(str))
    X_backtest[col] = le.transform(X_backtest[col].astype(str))

    encoders[col] = le

print("\nLabel Encoding completado.")

# ============================================================
# 14. VERIFICACIÓN FINAL
# ============================================================

print("\nTipos de datos finales:")
print(X_train.dtypes.value_counts())

print("\n¿Hay nulos en X_train?")
print(X_train.isnull().sum().sum())

print("\n¿Hay nulos en y_train?")
print(y_train.isnull().sum())
# ============================================================
# 15. IMPUTAR NULOS EN VARIABLES EXPLICATIVAS
# Mediana calculada SOLO con train para evitar leakage
# ============================================================

train_medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)
X_validation = X_validation.fillna(train_medians)
X_backtest = X_backtest.fillna(train_medians)

print("\nImputación de nulos en X completada.")
print("Nulos en X_train:", X_train.isnull().sum().sum())
print("Nulos en X_test:", X_test.isnull().sum().sum())
print("Nulos en X_validation:", X_validation.isnull().sum().sum())
print("Nulos en X_backtest:", X_backtest.isnull().sum().sum())
# ============================================================
# 16. ENTRENAR MODELO XGBOOST BASELINE
# ============================================================

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# ============================================================
# 17. PREDICCIONES EN TEST
# ============================================================

y_pred_test = xgb_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("\n--- RESULTADOS XGBOOST BASELINE EN TEST ---")
print("MAE:", round(mae_test, 4))
print("RMSE:", round(rmse_test, 4))
print("R²:", round(r2_test, 4))
print("\nResumen del target:")
print(y_train.describe())
print(y_test.describe())
# ============================================================
# 18. EVALUACIÓN EN VALIDATION
# ============================================================

y_pred_validation = xgb_model.predict(X_validation)

mae_val = mean_absolute_error(y_validation, y_pred_validation)
rmse_val = np.sqrt(mean_squared_error(y_validation, y_pred_validation))
r2_val = r2_score(y_validation, y_pred_validation)

print("\n--- RESULTADOS XGBOOST BASELINE EN VALIDATION ---")
print("MAE:", round(mae_val, 4))
print("RMSE:", round(rmse_val, 4))
print("R²:", round(r2_val, 4))

# ============================================================
# 19. GUARDAR MÉTRICAS
# ============================================================

metrics_xgb = pd.DataFrame({
    "modelo": ["XGBoost Baseline", "XGBoost Baseline"],
    "dataset": ["test", "validation"],
    "MAE": [mae_test, mae_val],
    "RMSE": [rmse_test, rmse_val],
    "R2": [r2_test, r2_val]
})

metrics_xgb.to_csv("outputs/tables/xgboost_baseline_metrics.csv", index=False)

print("\nMétricas guardadas en:")
print("outputs/tables/xgboost_baseline_metrics.csv")

# ============================================================
# 20. GUARDAR PREDICCIONES TEST Y VALIDATION
# ============================================================

pred_test = df_test.loc[y_test.index, ["month", "asin", "segment_seed", "category_seed"]].copy()
pred_test["dataset"] = "test"
pred_test["y_real_log"] = y_test.values
pred_test["y_pred_log"] = y_pred_test

pred_validation = df_validation.loc[y_validation.index, ["month", "asin", "segment_seed", "category_seed"]].copy()
pred_validation["dataset"] = "validation"
pred_validation["y_real_log"] = y_validation.values
pred_validation["y_pred_log"] = y_pred_validation

predictions_xgb = pd.concat([pred_test, pred_validation], axis=0)

predictions_xgb.to_csv("outputs/tables/xgboost_baseline_predictions.csv", index=False)

print("\nPredicciones guardadas en:")
print("outputs/tables/xgboost_baseline_predictions.csv")

# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance.to_csv("outputs/tables/xgboost_feature_importance.csv", index=False)

print("\nTop 15 variables más importantes:")
print(feature_importance.head(15))

print("\nFeature importance guardado en:")
print("outputs/tables/xgboost_feature_importance.csv")
# ============================================================
# 22. GRÁFICO FEATURE IMPORTANCE - TOP 15
# ============================================================

import matplotlib.pyplot as plt

top_features = feature_importance.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("XGBoost - Top 15 variables más importantes")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()

plt.savefig("outputs/figures/xgboost_feature_importance_top15.png", dpi=300)
plt.show()

print("\nGráfico guardado en:")
print("outputs/figures/xgboost_feature_importance_top15.png")
# ============================================================
# 23. GRÁFICO REAL VS PREDICHO - VALIDATION
# ============================================================

plt.figure(figsize=(8, 8))
plt.scatter(y_validation, y_pred_validation, alpha=0.3)
plt.plot(
    [y_validation.min(), y_validation.max()],
    [y_validation.min(), y_validation.max()],
    linestyle="--"
)
plt.title("XGBoost - Real vs Predicho en Validation")
plt.xlabel("Precio real log")
plt.ylabel("Precio predicho log")
plt.tight_layout()

plt.savefig("outputs/figures/xgboost_real_vs_pred_validation.png", dpi=300)
plt.show()

print("\nGráfico guardado en:")
print("outputs/figures/xgboost_real_vs_pred_validation.png")