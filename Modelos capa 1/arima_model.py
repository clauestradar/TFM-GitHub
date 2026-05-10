import pandas as pd
import numpy as np

df = pd.read_csv("analytic_panel_final_FINAL.csv")

# Limpiar nombres por si acaso
df.columns = df.columns.str.strip()

# Fecha real del dataset
df["month"] = pd.to_datetime(df["month"])

# Orden cronológico
df = df.sort_values("month").reset_index(drop=True)

# Fechas únicas mensuales
fechas = sorted(df["month"].unique())

# Corte 90% / 10% por meses
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

monthly_arimax = (
    df_modeling
    .groupby("month")
    .agg(
        y=("price_log", "mean"),
        cpi=("cpi", "mean"),
        cpi_beauty=("cpi_beauty", "mean"),
        unemployment_rate=("unemployment_rate", "mean"),
        interest_rate=("interest_rate", "mean"),
        inflation_yoy=("inflation_yoy", "mean"),
        real_interest_rate=("real_interest_rate", "mean")
    )
    .reset_index()
)

monthly_arimax = monthly_arimax.sort_values("month").set_index("month")

print(monthly_arimax.head())
print(monthly_arimax.isnull().sum())

# Variable objetivo
y = monthly_arimax["y"]

# Variables exógenas
X = monthly_arimax[
    [
        "cpi_beauty",
        "unemployment_rate",
        "interest_rate",
    ]
]
split_arimax = int(len(monthly_arimax) * 0.80)

y_train = y.iloc[:split_arimax]
y_test = y.iloc[split_arimax:]

X_train = X.iloc[:split_arimax]
X_test = X.iloc[split_arimax:]

print("Train:")
print(y_train.index.min(), "→", y_train.index.max())

print("\nTest:")
print(y_test.index.min(), "→", y_test.index.max())

#Entrenar Arimax
from statsmodels.tsa.statespace.sarimax import SARIMAX

arimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 0),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

arimax_result = arimax_model.fit(disp=False)

print(arimax_result.summary())

#Predicciones
forecast_arimax = arimax_result.get_forecast(
    steps=len(y_test),
    exog=X_test
)

y_pred = forecast_arimax.predicted_mean

conf_int = forecast_arimax.conf_int()

#Métricas
import numpy as np

# MAE
mae = np.mean(np.abs(y_test - y_pred))

# RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

# MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("ARIMAX Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

#Visualización
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# Train
plt.plot(
    y_train.index,
    y_train,
    label="Train",
    color="#0d3b66",   # azul oscuro
    linewidth=2
)

# Test real
plt.plot(
    y_test.index,
    y_test,
    label="Real Test",
    color="#ff4fa3",   # rosado
    linewidth=2
)

# Forecast
plt.plot(
    y_pred.index,
    y_pred,
    label="ARIMAX Forecast",
    color="#4ea8de",   # azul claro
    linewidth=2,
    linestyle="--"
)

# Intervalo confianza
plt.fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="#ffb3d9",   # rosado suave
    alpha=0.25
)

plt.title(
    "ARIMAX Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Log Precio", fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()