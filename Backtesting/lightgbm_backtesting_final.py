# ============================================================
# LIGHTGBM - BACKTESTING FINAL TFM
# Usa el 10% final del dataset como bloque temporal futuro
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paleta visual del TFM
COLOR_AZUL = "#1f77b4"
COLOR_ROSADO = "#d45087"
COLOR_ROSADO_SUAVE = "#f4a6c1"

from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CARGAR DATASET
# ============================================================

df = pd.read_csv("analytic_panel_final_FINAL.csv")

df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["month", "asin"]).reset_index(drop=True)

print("Shape dataset:", df.shape)
print("Fecha mínima:", df["month"].min())
print("Fecha máxima:", df["month"].max())


# ============================================================
# 2. CREAR LAGS Y ROLLING MEANS
# ============================================================

df = df.sort_values(["asin", "month"]).reset_index(drop=True)

df["price_log_lag_1"] = df.groupby("asin")["price_log"].shift(1)
df["price_log_lag_12"] = df.groupby("asin")["price_log"].shift(12)

df["price_log_roll_mean_3"] = (
    df.groupby("asin")["price_log"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["price_log_roll_mean_12"] = (
    df.groupby("asin")["price_log"]
    .shift(1)
    .rolling(12)
    .mean()
    .reset_index(level=0, drop=True)
)

macro_cols = [
    "cpi_beauty",
    "unemployment_rate",
    "interest_rate",
    "inflation_yoy"
]

for col in macro_cols:
    if col in df.columns:
        df[f"{col}_lag_1"] = df.groupby("asin")[col].shift(1)
        df[f"{col}_lag_12"] = df.groupby("asin")[col].shift(12)

print("Lags creados correctamente.")


# ============================================================
# 3. SPLIT TEMPORAL: 90% MODELADO / 10% BACKTESTING
# ============================================================

months = np.array(sorted(df["month"].unique()))
n_months = len(months)

n_backtest_months = int(n_months * 0.10)

backtest_months = months[-n_backtest_months:]
modeling_months = months[:-n_backtest_months]

df_modeling = df[df["month"].isin(modeling_months)].copy()
df_backtest = df[df["month"].isin(backtest_months)].copy()

print("\nModeling:")
print(df_modeling["month"].min(), "→", df_modeling["month"].max())
print("Meses:", df_modeling["month"].nunique())

print("\nBacktesting:")
print(df_backtest["month"].min(), "→", df_backtest["month"].max())
print("Meses:", df_backtest["month"].nunique())


# ============================================================
# 4. DEFINIR TARGET Y COLUMNAS A ELIMINAR
# ============================================================

target = "price_log"

columns_to_drop = [
    target,
    "month",
    "asin",
    "segment_seed",
    "coverage_pct",
    "max_null_run",
    "low_quality_product",
    "was_imputed",
    "was_null",
    "amazon_price",
    "amazon_price_raw",
    "amazon_price_final",
    "list_price",
    "list_price_raw",
    "list_price_final",
    "price_change_pct",
    "discount_rate",
    "category_mean_price",
    "price_vs_category_mean",
    "real_price",
    "demand_proxy",
    "sales_rank_raw",
    "sales_rank_log",
    "sales_rank_change",
    "amazon_price_imputed",
    "product_title"
]

columns_to_drop_existing = [
    col for col in columns_to_drop if col in df.columns
]


# ============================================================
# 5. CREAR X / y
# ============================================================

X_modeling = df_modeling.drop(columns=columns_to_drop_existing)
y_modeling = df_modeling[target]

X_backtest = df_backtest.drop(columns=columns_to_drop_existing)
y_backtest = df_backtest[target]

modeling_mask = y_modeling.notnull()
backtest_mask = y_backtest.notnull()

X_modeling = X_modeling[modeling_mask].copy()
y_modeling = y_modeling[modeling_mask].copy()

X_backtest = X_backtest[backtest_mask].copy()
y_backtest = y_backtest[backtest_mask].copy()

print("\nShapes finales:")
print("X_modeling:", X_modeling.shape)
print("X_backtest:", X_backtest.shape)


# ============================================================
# 6. LABEL ENCODING
# ============================================================

categorical_cols = X_modeling.select_dtypes(
    include=["object", "category"]
).columns.tolist()

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()

    combined_values = pd.concat([
        X_modeling[col],
        X_backtest[col]
    ]).astype(str)

    le.fit(combined_values)

    X_modeling[col] = le.transform(X_modeling[col].astype(str))
    X_backtest[col] = le.transform(X_backtest[col].astype(str))

    encoders[col] = le

print("\nLabel Encoding completado.")
print("Variables categóricas:", categorical_cols)


# ============================================================
# 7. IMPUTAR NULOS
# Mediana calculada solo con modeling
# ============================================================

modeling_medians = X_modeling.median(numeric_only=True)

X_modeling = X_modeling.fillna(modeling_medians)
X_backtest = X_backtest.fillna(modeling_medians)

print("\nNulos X_modeling:", X_modeling.isnull().sum().sum())
print("Nulos X_backtest:", X_backtest.isnull().sum().sum())


# ============================================================
# 8. ENTRENAR LIGHTGBM GANADOR
# ============================================================

best_lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

best_lgbm_model.fit(X_modeling, y_modeling)

print("\nModelo LightGBM entrenado con el 90% inicial.")


# ============================================================
# 9. BACKTESTING SOBRE 10% FINAL
# ============================================================

y_pred_backtest = best_lgbm_model.predict(X_backtest)


def calcular_mape(y_real, y_pred):
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100


backtest_metrics = {
    "Fase": "Backtesting",
    "MAE": mean_absolute_error(y_backtest, y_pred_backtest),
    "RMSE": np.sqrt(mean_squared_error(y_backtest, y_pred_backtest)),
    "MAPE": calcular_mape(y_backtest, y_pred_backtest),
    "R2": r2_score(y_backtest, y_pred_backtest),
    "Accuracy_Forecast": 100 - calcular_mape(y_backtest, y_pred_backtest)
}

backtest_metrics_df = pd.DataFrame([backtest_metrics])

print("\nRESULTADOS BACKTESTING LIGHTGBM")
print(backtest_metrics_df)

backtest_metrics_df.to_csv(
    "lightgbm_backtesting_metrics.csv",
    index=False
)


# ============================================================
# 10. GUARDAR PREDICCIONES
# ============================================================

backtest_results = df_backtest.loc[y_backtest.index, ["month", "asin"]].copy()
backtest_results["y_real"] = y_backtest.values
backtest_results["y_pred"] = y_pred_backtest
backtest_results["error"] = backtest_results["y_real"] - backtest_results["y_pred"]
backtest_results["abs_error"] = np.abs(backtest_results["error"])
backtest_results["ape"] = (
    np.abs(backtest_results["error"] / backtest_results["y_real"]) * 100
)

backtest_results.to_csv(
    "lightgbm_backtesting_predictions.csv",
    index=False
)

print("\nPredicciones guardadas como lightgbm_backtesting_predictions.csv")


# ============================================================
# 11. MÉTRICAS MENSUALES
# ============================================================

monthly_backtest = (
    backtest_results
    .groupby("month")
    .apply(lambda x: pd.Series({
        "MAE": mean_absolute_error(x["y_real"], x["y_pred"]),
        "RMSE": np.sqrt(mean_squared_error(x["y_real"], x["y_pred"])),
        "MAPE": calcular_mape(x["y_real"], x["y_pred"]),
        "Accuracy_Forecast": 100 - calcular_mape(x["y_real"], x["y_pred"])
    }))
    .reset_index()
)

monthly_backtest.to_csv(
    "lightgbm_backtesting_monthly_metrics.csv",
    index=False
)

print("\nMétricas mensuales:")
print(monthly_backtest)


# ============================================================
# 12. GRÁFICOS
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_backtest,
    y_pred_backtest,
    alpha=0.25,
    color=COLOR_AZUL,
    edgecolors="none"
)

plt.plot(
    [y_backtest.min(), y_backtest.max()],
    [y_backtest.min(), y_backtest.max()],
    linestyle="--",
    linewidth=2,
    color=COLOR_ROSADO
)

plt.title(
    "LightGBM - Backtesting: Real vs Predicho",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "lightgbm_backtesting_real_vs_predicho.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    monthly_backtest["month"],
    monthly_backtest["MAPE"],
    marker="o",
    linewidth=2,
    color=COLOR_ROSADO
)

plt.title(
    "LightGBM - MAPE mensual en Backtesting",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Mes")
plt.ylabel("MAPE (%)")
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "lightgbm_backtesting_mape_mensual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


monthly_real_pred = (
    backtest_results
    .groupby("month")[["y_real", "y_pred"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_real_pred["month"],
    monthly_real_pred["y_real"],
    marker="o",
    linewidth=2,
    color=COLOR_AZUL,
    label="Real"
)

plt.plot(
    monthly_real_pred["month"],
    monthly_real_pred["y_pred"],
    marker="o",
    linewidth=2,
    color=COLOR_ROSADO,
    label="Predicho"
)

plt.title(
    "LightGBM - Precio log promedio real vs predicho en Backtesting",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Mes")
plt.ylabel("price_log promedio")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "lightgbm_backtesting_real_vs_predicho_mensual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nBacktesting finalizado correctamente.")