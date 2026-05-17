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
# 7. GRID SEARCH ARIMAX
# ============================================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

results = []

# Órdenes ARIMA a probar
orders = [
    (0,1,0),
    (1,1,0),
    (0,1,1),
    (1,1,1),
    (2,1,0),
    (2,1,1)
]

for order in orders:
    
    maes = []
    rmses = []
    mapes = []
    
    print(f"\nEvaluando ARIMAX {order}")
    
    for train_idx, test_idx in tscv.split(X):
        
        # Split temporal
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        try:
            
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=order,
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            result = model.fit(disp=False)
            
            forecast = result.get_forecast(
                steps=len(y_test),
                exog=X_test
            )
            
            y_pred = forecast.predicted_mean
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            
            rmse = np.sqrt(
                mean_squared_error(y_test, y_pred)
            )
            
            mape = np.mean(
                np.abs((y_test - y_pred) / y_test)
            ) * 100
            
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            
        except Exception as e:
            
            print(f"Error con {order}: {e}")
    
    # Guardar promedio
    results.append({
        "order": order,
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "MAPE": np.mean(mapes)
    })

# Resultados finales
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAPE")

print("\nRESULTADOS GRID SEARCH")
print(results_df)
# ============================================================
# 8. SELECCIÓN MEJOR MODELO
# ============================================================

best_order = results_df.iloc[0]["order"]

print("\nMEJOR MODELO ARIMAX:")
print(best_order)
# ============================================================
# 9. TRAIN / TEST FINAL INTERNO
# ============================================================

split_final = int(len(monthly_arimax) * 0.80)

train_final = monthly_arimax.iloc[:split_final]
test_final = monthly_arimax.iloc[split_final:]

# Variable objetivo
y_train_final = train_final["y"]
y_test_final = test_final["y"]

# Variables exógenas
X_train_final = train_final.drop(columns=["y"])
X_test_final = test_final.drop(columns=["y"])

print("\nTrain final:")
print(train_final.index.min(), "→", train_final.index.max())

print("\nTest final:")
print(test_final.index.min(), "→", test_final.index.max())
# ============================================================
# 10. ENTRENAR MEJOR ARIMAX
# ============================================================

best_model = SARIMAX(
    y_train_final,
    exog=X_train_final,
    order=best_order,
    seasonal_order=(0,0,0,0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

best_result = best_model.fit(disp=False)

print(best_result.summary())
# ============================================================
# 11. FORECAST FINAL INTERNO
# ============================================================

forecast_final = best_result.get_forecast(
    steps=len(y_test_final),
    exog=X_test_final
)

y_pred_final = forecast_final.predicted_mean

conf_int_final = forecast_final.conf_int()
# ============================================================
# 12. MÉTRICAS FINALES
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
    np.abs((y_test_final - y_pred_final) / y_test_final)
) * 100

print("\nARIMAX FINAL RESULTS")
print("MAE:", mae_final)
print("RMSE:", rmse_final)
print("MAPE:", mape_final)
# ============================================================
# 13. GRÁFICO FINAL
# ============================================================

plt.figure(figsize=(12,6))

# Train
plt.plot(
    y_train_final.index,
    y_train_final,
    label="Train",
    color="#0d3b66",
    linewidth=2
)

# Real
plt.plot(
    y_test_final.index,
    y_test_final,
    label="Real Test",
    color="#ff4fa3",
    linewidth=2
)

# Forecast
plt.plot(
    y_pred_final.index,
    y_pred_final,
    label="ARIMAX Forecast",
    color="#4ea8de",
    linewidth=2,
    linestyle="--"
)

# Intervalos
plt.fill_between(
    conf_int_final.index,
    conf_int_final.iloc[:,0],
    conf_int_final.iloc[:,1],
    color="#ffb3d9",
    alpha=0.25
)

plt.title(
    "ARIMAX Tuned Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Fecha")
plt.ylabel("Log Precio")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()