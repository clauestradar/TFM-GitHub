# ============================================================
# PROPHET - Capa 1 Baseline Temporal
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ============================================================
# 1. CARGAR DATOS
# ============================================================

df = pd.read_csv("analytic_panel_final_FINAL.csv")

df.columns = df.columns.str.strip()
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)

# ============================================================
# 2. DIVISIÓN TEMPORAL 90% / 10%
# ============================================================

fechas = sorted(df["month"].unique())

split_idx = int(len(fechas) * 0.90)

fechas_modeling = fechas[:split_idx]
fechas_backtest = fechas[split_idx:]

df_modeling = df[df["month"].isin(fechas_modeling)].copy()
df_backtest = df[df["month"].isin(fechas_backtest)].copy()

print("Total meses:", len(fechas))
print("Meses modelado:", len(fechas_modeling))
print("Meses backtesting reservado:", len(fechas_backtest))

print("\nRango modelado:")
print(df_modeling["month"].min(), "→", df_modeling["month"].max())

print("\nRango backtesting:")
print(df_backtest["month"].min(), "→", df_backtest["month"].max())

# ============================================================
# 3. CREAR SERIE MENSUAL AGREGADA
# ============================================================

monthly_prophet = (
    df_modeling
    .groupby("month")
    .agg(
        y=("price_log", "mean"),
        cpi_beauty=("cpi_beauty", "mean"),
        unemployment_rate=("unemployment_rate", "mean"),
        interest_rate=("interest_rate", "mean")
    )
    .reset_index()
)

monthly_prophet = monthly_prophet.sort_values("month")

# Prophet necesita columna ds
monthly_prophet = monthly_prophet.rename(columns={"month": "ds"})

print("\nPrimeras filas Prophet:")
print(monthly_prophet.head())

print("\nValores nulos:")
print(monthly_prophet.isnull().sum())

monthly_prophet = monthly_prophet.dropna()

# ============================================================
# 4. TRAIN / TEST INTERNO DENTRO DEL 90%
# ============================================================

split_prophet = int(len(monthly_prophet) * 0.80)

train_prophet = monthly_prophet.iloc[:split_prophet].copy()
test_prophet = monthly_prophet.iloc[split_prophet:].copy()

print("\nTrain:")
print(train_prophet["ds"].min(), "→", train_prophet["ds"].max())

print("\nTest:")
print(test_prophet["ds"].min(), "→", test_prophet["ds"].max())

# ============================================================
# 5. ENTRENAR MODELO PROPHET
# ============================================================

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="additive"
)

# Regresores macroeconómicos
prophet_model.add_regressor("cpi_beauty")
prophet_model.add_regressor("unemployment_rate")
prophet_model.add_regressor("interest_rate")

prophet_model.fit(train_prophet)

# ============================================================
# 6. FORECAST SOBRE TEST INTERNO
# ============================================================

future_prophet = test_prophet[
    [
        "ds",
        "cpi_beauty",
        "unemployment_rate",
        "interest_rate"
    ]
].copy()

forecast_prophet = prophet_model.predict(future_prophet)

y_test = test_prophet["y"].values
y_pred_prophet = forecast_prophet["yhat"].values

# ============================================================
# 7. MÉTRICAS MANUALES
# ============================================================

mae_prophet = np.mean(np.abs(y_test - y_pred_prophet))

rmse_prophet = np.sqrt(
    np.mean((y_test - y_pred_prophet) ** 2)
)

mape_prophet = np.mean(
    np.abs((y_test - y_pred_prophet) / y_test)
) * 100

print("\nPROPHET Results")
print("MAE:", mae_prophet)
print("RMSE:", rmse_prophet)
print("MAPE:", mape_prophet)

# ============================================================
# 8. GRÁFICO PROPHET
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    train_prophet["ds"],
    train_prophet["y"],
    label="Train",
    color="#0d3b66",
    linewidth=2
)

plt.plot(
    test_prophet["ds"],
    test_prophet["y"],
    label="Real Test",
    color="#ff4fa3",
    linewidth=2
)

plt.plot(
    forecast_prophet["ds"],
    forecast_prophet["yhat"],
    label="Prophet Forecast",
    color="#4ea8de",
    linewidth=2,
    linestyle="--"
)

plt.fill_between(
    forecast_prophet["ds"],
    forecast_prophet["yhat_lower"],
    forecast_prophet["yhat_upper"],
    color="#ffb3d9",
    alpha=0.25
)

plt.title(
    "Prophet Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Log Precio", fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()