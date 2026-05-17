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

print(monthly_arimax.head())
print(monthly_arimax.isnull().sum())
# ============================================================
# 4. CREAR LAGS TEMPORALES
# ============================================================

# Lags variable dependiente
monthly_arimax["y_lag_1"] = monthly_arimax["y"].shift(1)

monthly_arimax["y_lag_12"] = monthly_arimax["y"].shift(12)

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
# ============================================================
# 6. TIM SERIES SPLIT
# ============================================================

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

print("\nTimeSeriesSplit configurado:")
print(tscv)
# ============================================================
# 7. PREPARAR DATASET PARA PROPHET
# ============================================================

# Reset index
monthly_prophet = monthly_arimax.reset_index()

# Prophet requiere columnas:
# ds -> fecha
# y -> target

monthly_prophet = monthly_prophet.rename(
    columns={
        "month": "ds"
    }
)

print("\nDataset Prophet:")
print(monthly_prophet.head())
# ============================================================
# 8. VARIABLES PARA PROPHET
# ============================================================

# Variable objetivo
y = monthly_prophet["y"]

# Variables explicativas
X = monthly_prophet.drop(columns=["y"])

print("\nVariables Prophet:")
print(X.columns.tolist())
# ============================================================
# 9. TIME SERIES SPLIT
# ============================================================

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

print("\nTimeSeriesSplit Prophet:")
print(tscv)
# ============================================================
# 10. GRID SEARCH PROPHET
# ============================================================

from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

results = []

# Hiperparámetros Prophet
changepoint_scales = [0.01, 0.05, 0.1]

seasonality_modes = [
    "additive",
    "multiplicative"
]

seasonality_prior_scales = [1.0, 5.0]

for cps in changepoint_scales:
    
    for smode in seasonality_modes:
        
        for sps in seasonality_prior_scales:
            
            maes = []
            rmses = []
            mapes = []
            
            print(
                f"\nEvaluando Prophet | "
                f"cps={cps} | "
                f"mode={smode} | "
                f"sps={sps}"
            )
            
            for train_idx, test_idx in tscv.split(monthly_prophet):
                
                train_fold = monthly_prophet.iloc[train_idx]
                test_fold = monthly_prophet.iloc[test_idx]
                
                try:
                    
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=cps,
                        seasonality_mode=smode,
                        seasonality_prior_scale=sps
                    )
                    
                    # Agregar regresores
                    regressors = [
                        col for col in monthly_prophet.columns
                        if col not in ["ds", "y"]
                    ]
                    
                    for reg in regressors:
                        model.add_regressor(reg)
                    
                    model.fit(train_fold)
                    
                    future = test_fold.drop(columns=["y"])
                    
                    forecast = model.predict(future)
                    
                    y_test = test_fold["y"].values
                    y_pred = forecast["yhat"].values
                    
                    # Métricas
                    mae = mean_absolute_error(
                        y_test,
                        y_pred
                    )
                    
                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test,
                            y_pred
                        )
                    )
                    
                    mape = np.mean(
                        np.abs(
                            (y_test - y_pred) / y_test
                        )
                    ) * 100
                    
                    maes.append(mae)
                    rmses.append(rmse)
                    mapes.append(mape)
                    
                except Exception as e:
                    
                    print(
                        f"Error Prophet: {e}"
                    )
            
            if len(mapes) > 0:
                
                results.append({
                    "changepoint_prior_scale": cps,
                    "seasonality_mode": smode,
                    "seasonality_prior_scale": sps,
                    "MAE": np.mean(maes),
                    "RMSE": np.mean(rmses),
                    "MAPE": np.mean(mapes)
                })

# Resultados finales
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAPE")

print("\nRESULTADOS GRID SEARCH PROPHET")
print(results_df)
# ============================================================
# 11. SELECCIÓN MEJOR MODELO PROPHET
# ============================================================

best_cps = results_df.iloc[0]["changepoint_prior_scale"]

best_mode = results_df.iloc[0]["seasonality_mode"]

best_sps = results_df.iloc[0]["seasonality_prior_scale"]

print("\nMEJOR MODELO PROPHET")
print("changepoint_prior_scale:", best_cps)
print("seasonality_mode:", best_mode)
print("seasonality_prior_scale:", best_sps)
# ============================================================
# 12. TRAIN / TEST FINAL INTERNO
# ============================================================

split_final = int(len(monthly_prophet) * 0.80)

train_final = monthly_prophet.iloc[:split_final].copy()

test_final = monthly_prophet.iloc[split_final:].copy()

print("\nTrain final:")
print(train_final["ds"].min(), "→", train_final["ds"].max())

print("\nTest final:")
print(test_final["ds"].min(), "→", test_final["ds"].max())
# ============================================================
# 13. ENTRENAR MEJOR PROPHET
# ============================================================

best_prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=best_cps,
    seasonality_mode=best_mode,
    seasonality_prior_scale=best_sps
)

# Regresores
regressors = [
    col for col in monthly_prophet.columns
    if col not in ["ds", "y"]
]

for reg in regressors:
    best_prophet_model.add_regressor(reg)

best_prophet_model.fit(train_final)
# ============================================================
# 14. FORECAST FINAL INTERNO
# ============================================================

future_final = test_final.drop(columns=["y"])

forecast_final = best_prophet_model.predict(
    future_final
)

y_test_final = test_final["y"].values

y_pred_final = forecast_final["yhat"].values
# ============================================================
# 15. MÉTRICAS FINALES PROPHET
# ============================================================

mae_final = mean_absolute_error(
    y_test_final,
    y_pred_final
)

rmse_final = np.sqrt(
    mean_squared_error(
        y_test_final,
        y_pred_final
    )
)

mape_final = np.mean(
    np.abs(
        (y_test_final - y_pred_final)
        / y_test_final
    )
) * 100

print("\nPROPHET FINAL RESULTS")
print("MAE:", mae_final)
print("RMSE:", rmse_final)
print("MAPE:", mape_final)
# ============================================================
# 16. GRÁFICO FINAL PROPHET
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# Train
plt.plot(
    train_final["ds"],
    train_final["y"],
    label="Train",
    color="#0d3b66",
    linewidth=2
)

# Real
plt.plot(
    test_final["ds"],
    test_final["y"],
    label="Real Test",
    color="#ff4fa3",
    linewidth=2
)

# Forecast
plt.plot(
    forecast_final["ds"],
    forecast_final["yhat"],
    label="Prophet Forecast",
    color="#4ea8de",
    linewidth=2,
    linestyle="--"
)

# Intervalos
plt.fill_between(
    forecast_final["ds"],
    forecast_final["yhat_lower"],
    forecast_final["yhat_upper"],
    color="#ffb3d9",
    alpha=0.25
)

plt.title(
    "Prophet Tuned Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha")
plt.ylabel("Log Precio")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()