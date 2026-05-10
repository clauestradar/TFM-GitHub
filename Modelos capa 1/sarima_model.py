# ============================================================
# SARIMAX - Capa 1 Baseline Temporal
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================
# 1. CARGAR DATOS
# ============================================================

df = pd.read_csv("analytic_panel_final_FINAL.csv")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Convertir fecha
df["month"] = pd.to_datetime(df["month"])

# Orden cronológico
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

monthly_sarimax = (
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

monthly_sarimax = monthly_sarimax.sort_values("month").set_index("month")

# Forzar frecuencia mensual
monthly_sarimax = monthly_sarimax.asfreq("MS")

print("\nPrimeras filas:")
print(monthly_sarimax.head())

print("\nValores nulos:")
print(monthly_sarimax.isnull().sum())

# Eliminar nulos si existieran
monthly_sarimax = monthly_sarimax.dropna()

# ============================================================
# 4. DEFINIR VARIABLE OBJETIVO Y EXÓGENAS
# ============================================================

y = monthly_sarimax["y"]

X = monthly_sarimax[
    [
        "cpi_beauty",
        "unemployment_rate",
        "interest_rate"
    ]
]

# ============================================================
# 5. TRAIN / TEST INTERNO DENTRO DEL 90%
# ============================================================

split_sarimax = int(len(monthly_sarimax) * 0.80)

y_train = y.iloc[:split_sarimax]
y_test = y.iloc[split_sarimax:]

X_train = X.iloc[:split_sarimax]
X_test = X.iloc[split_sarimax:]

print("\nTrain:")
print(y_train.index.min(), "→", y_train.index.max())

print("\nTest:")
print(y_test.index.min(), "→", y_test.index.max())

# ============================================================
# 6. ENTRENAR MODELO SARIMAX
# ============================================================

sarimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 0),
    seasonal_order=(1, 0, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_result = sarimax_model.fit(disp=False)

print("\nSARIMAX Summary")
print(sarimax_result.summary())

# ============================================================
# 7. FORECAST SOBRE TEST INTERNO
# ============================================================

forecast_sarimax = sarimax_result.get_forecast(
    steps=len(y_test),
    exog=X_test
)

y_pred_sarimax = forecast_sarimax.predicted_mean
conf_int_sarimax = forecast_sarimax.conf_int()

# ============================================================
# 8. MÉTRICAS MANUALES
# ============================================================

mae_sarimax = np.mean(np.abs(y_test - y_pred_sarimax))

rmse_sarimax = np.sqrt(
    np.mean((y_test - y_pred_sarimax) ** 2)
)

mape_sarimax = np.mean(
    np.abs((y_test - y_pred_sarimax) / y_test)
) * 100

print("\nSARIMAX Results")
print("MAE:", mae_sarimax)
print("RMSE:", rmse_sarimax)
print("MAPE:", mape_sarimax)

# ============================================================
# 9. GRÁFICO SARIMAX
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_train.index,
    y_train,
    label="Train",
    color="#0d3b66",
    linewidth=2
)

plt.plot(
    y_test.index,
    y_test,
    label="Real Test",
    color="#ff4fa3",
    linewidth=2
)

plt.plot(
    y_pred_sarimax.index,
    y_pred_sarimax,
    label="SARIMAX Forecast",
    color="#4ea8de",
    linewidth=2,
    linestyle="--"
)

plt.fill_between(
    conf_int_sarimax.index,
    conf_int_sarimax.iloc[:, 0],
    conf_int_sarimax.iloc[:, 1],
    color="#ffb3d9",
    alpha=0.25
)

plt.title(
    "SARIMAX Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Log Precio", fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()