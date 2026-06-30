import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

print("\nDataset mensual inicial:")
print(monthly_arimax.head())
print("\nValores nulos:")
print(monthly_arimax.isnull().sum())
# ============================================================
# 4. CREAR LAGS TEMPORALES
# ============================================================

# Lags variables exógenas
for col in [
    "cpi_beauty",
    "unemployment_rate",
    "interest_rate"
]:
    
    monthly_arimax[f"{col}_lag_12"] = (
        monthly_arimax[col].shift(12)
    )

# Eliminar nulos
monthly_arimax = monthly_arimax.dropna()

print("\nDataset con lags:")
print(monthly_arimax.head())

print("\nShape final:")
print(monthly_arimax.shape)
# ============================================================
# 5. DEFINIR VARIABLES
# ============================================================
# Variable objetivo
y = monthly_arimax["y"]

# Variables explicativas
X = monthly_arimax.drop(columns=["y"])

print("\nVariables exógenas:")
print(X.columns.tolist())

def mape_score(y_true, y_pred):
    """Calcula MAPE evitando problemas si hubiese valores cero."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_metrics(y_true, y_pred):
    """Devuelve MAE, MAPE y RMSE."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }
# ============================================================
# 6. TIM SERIES SPLIT
# ============================================================

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

print("\nTimeSeriesSplit configurado:")
print(tscv)
# ============================================================
# 7. GRID SEARCH SARIMAX
# ============================================================

results = []

# Órdenes ARIMA
orders = [
    (0,1,0),
    (1,1,0),
    (0,1,1),
    (1,1,1),
    (2,1,0)
]

# Órdenes estacionales
seasonal_orders = [
    (0,0,0,12),
    (1,0,0,12),
    (0,1,0,12),
    (1,1,0,12)
]

# Tendencias
trends = ["n", "c", "t", "ct"]

for order in orders:

    for seasonal_order in seasonal_orders:

        for trend in trends:

            maes_train = []
            mapes_train = []
            rmses_train = []

            maes_test = []
            mapes_test = []
            rmses_test = []

            print(
                f"\nEvaluando SARIMAX "
                f"{order} x {seasonal_order} "
                f"| trend={trend}"
            )

            for train_idx, test_idx in tscv.split(X):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                try:

                    model = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    result = model.fit(disp=False)

                    # TRAIN
                    y_pred_train = result.fittedvalues

                    # TEST
                    forecast = result.get_forecast(
                        steps=len(y_test),
                        exog=X_test
                    )

                    y_pred_test = forecast.predicted_mean

                    # ====================
                    # TRAIN METRICS
                    # ====================

                    train_metrics = evaluate_metrics(
                        y_train,
                        y_pred_train
                    )

                    maes_train.append(train_metrics["MAE"])
                    mapes_train.append(train_metrics["MAPE"])
                    rmses_train.append(train_metrics["RMSE"])

                    # ====================
                    # TEST METRICS
                    # ====================

                    test_metrics = evaluate_metrics(
                        y_test,
                        y_pred_test
                    )

                    maes_test.append(test_metrics["MAE"])
                    mapes_test.append(test_metrics["MAPE"])
                    rmses_test.append(test_metrics["RMSE"])

                except Exception as e:

                    print(
                        f"Error con "
                        f"{order} x {seasonal_order} "
                        f"| trend={trend}: {e}"
                    )

            if len(mapes_test) > 0:

                results.append({

                    "order": order,
                    "seasonal_order": seasonal_order,
                    "trend": trend,

                    "MAE_train": np.mean(maes_train),
                    "MAPE_train": np.mean(mapes_train),
                    "RMSE_train": np.mean(rmses_train),

                    "MAE_test": np.mean(maes_test),
                    "MAPE_test": np.mean(mapes_test),
                    "RMSE_test": np.mean(rmses_test)

                })

# Resultados
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAPE_test"
).reset_index(drop=True)

print("\nRESULTADOS GRID SEARCH SARIMAX")
print(results_df)

# Guardar tabla
results_df.to_csv(
    "sarimax_grid_search_results.csv",
    index=False
)

print(
    "\nTabla guardada como: sarimax_grid_search_results.csv"
)
# ============================================================
# 8. SELECCIÓN MEJOR MODELO SARIMAX
# ============================================================

best_order = results_df.iloc[0]["order"]

best_seasonal_order = results_df.iloc[0]["seasonal_order"]

best_trend = results_df.iloc[0]["trend"]

print("\nMEJOR MODELO SARIMAX:")
print("order:", best_order)
print("seasonal_order:", best_seasonal_order)
print("trend:", best_trend)
# ============================================================
# 9. TRAIN / TEST FINAL INTERNO
# ============================================================

split_final = int(len(monthly_arimax) * 0.80)

train_final = monthly_arimax.iloc[:split_final]
test_final = monthly_arimax.iloc[split_final:]

y_train_final = train_final["y"]
y_test_final = test_final["y"]

X_train_final = train_final.drop(columns=["y"])
X_test_final = test_final.drop(columns=["y"])

print("\nTrain final:")
print(train_final.index.min(), "→", train_final.index.max())

print("\nTest final:")
print(test_final.index.min(), "→", test_final.index.max())
# ============================================================
# 10. ENTRENAR MEJOR SARIMAX
# ============================================================

best_sarimax_model = SARIMAX(
    y_train_final,
    exog=X_train_final,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
    trend=best_trend,
)

best_sarimax_result = best_sarimax_model.fit(disp=False)

print(best_sarimax_result.summary())
# ============================================================
# 11. FORECAST FINAL INTERNO
# ============================================================

forecast_sarimax_final = best_sarimax_result.get_forecast(
    steps=len(y_test_final),
    exog=X_test_final
)

y_pred_sarimax_final = forecast_sarimax_final.predicted_mean
conf_int_sarimax_final = forecast_sarimax_final.conf_int()
# ============================================================
# 12. MÉTRICAS FINALES SARIMAX
# ============================================================
# Predicción TRAIN
y_pred_train_final = best_sarimax_result.fittedvalues

# Métricas TRAIN
metrics_train_final = evaluate_metrics(
    y_train_final,
    y_pred_train_final
)

# Métricas TEST
metrics_test_final = evaluate_metrics(
    y_test_final,
    y_pred_sarimax_final
)

# Tabla final
final_metrics_df = pd.DataFrame({

    "dataset": ["train", "test"],

    "MAE": [
        metrics_train_final["MAE"],
        metrics_test_final["MAE"]
    ],

    "MAPE": [
        metrics_train_final["MAPE"],
        metrics_test_final["MAPE"]
    ],

    "RMSE": [
        metrics_train_final["RMSE"],
        metrics_test_final["RMSE"]
    ]

})

print("\nSARIMAX FINAL RESULTS")
print(final_metrics_df)

# Guardar métricas
final_metrics_df.to_csv(
    "sarimax_final_train_test_metrics.csv",
    index=False
)

print(
    "\nMétricas guardadas como:"
    " sarimax_final_train_test_metrics.csv"
)
# ============================================================
# 13. GRÁFICO FINAL SARIMAX
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(
    y_train_final.index,
    y_train_final,
    label="Train",
    color="#0d3b66",
    linewidth=2
)

plt.plot(
    y_test_final.index,
    y_test_final,
    label="Real Test",
    color="#ff4fa3",
    linewidth=2
)

plt.plot(
    y_pred_sarimax_final.index,
    y_pred_sarimax_final,
    label="SARIMAX Forecast",
    color="#4ea8de",
    linewidth=2,
    linestyle="--"
)

plt.fill_between(
    conf_int_sarimax_final.index,
    conf_int_sarimax_final.iloc[:, 0],
    conf_int_sarimax_final.iloc[:, 1],
    color="#ffb3d9",
    alpha=0.25
)

plt.title(
    "SARIMAX Tuned Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha")
plt.ylabel("Log Precio")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()