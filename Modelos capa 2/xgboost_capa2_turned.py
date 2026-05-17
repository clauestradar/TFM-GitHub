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
# 6.1 CREAR LAGS TEMPORALES
# ============================================================

df = df.sort_values(["asin", "month"]).reset_index(drop=True)

# Lags de la variable dependiente
df["price_log_lag_1"] = (
    df.groupby("asin")["price_log"].shift(1)
)

df["price_log_lag_12"] = (
    df.groupby("asin")["price_log"].shift(12)
)

# Rolling means usando solo pasado
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

# Variables macro con lags
macro_cols = [
    "cpi_beauty",
    "unemployment_rate",
    "interest_rate",
    "inflation_yoy"
]

for col in macro_cols:

    if col in df.columns:

        df[f"{col}_lag_1"] = (
            df.groupby("asin")[col].shift(1)
        )

        df[f"{col}_lag_12"] = (
            df.groupby("asin")[col].shift(12)
        )

print("\nLags creados correctamente.")
print("Shape dataset con lags:", df.shape)
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
    "segment_seed",

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
# 16. GRID SEARCH XGBOOST
# ============================================================

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

results = []

learning_rates = [0.01, 0.05, 0.10]
max_depths = [3, 5, 7]
subsamples = [0.8, 1.0]
colsamples = [0.8, 1.0]

for lr in learning_rates:
    
    for depth in max_depths:
        
        for subs in subsamples:
            
            for cols in colsamples:
                
                print(
                    f"\nEvaluando:"
                    f" lr={lr}"
                    f" depth={depth}"
                    f" subs={subs}"
                    f" cols={cols}"
                )
                
                model = XGBRegressor(
                    n_estimators=500,
                    learning_rate=lr,
                    max_depth=depth,
                    subsample=subs,
                    colsample_bytree=cols,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(
                    X_train,
                    y_train
                )
                
                # Predicción validation
                y_pred_val = model.predict(
                    X_validation
                )
                
                mae = mean_absolute_error(
                    y_validation,
                    y_pred_val
                )
                
                rmse = np.sqrt(
                    mean_squared_error(
                        y_validation,
                        y_pred_val
                    )
                )
                
                r2 = r2_score(
                    y_validation,
                    y_pred_val
                )
                
                results.append({
                    "learning_rate": lr,
                    "max_depth": depth,
                    "subsample": subs,
                    "colsample_bytree": cols,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
)

print("\nRESULTADOS GRID SEARCH XGBOOST")
print(results_df.head(10))
# ============================================================
# 17. SELECCIÓN MEJOR MODELO XGBOOST
# ============================================================

best_params = results_df.iloc[0]

print("\nMEJOR MODELO XGBOOST")
print(best_params)
# ============================================================
# 18. ENTRENAR MEJOR XGBOOST
# ============================================================

best_xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=best_params["learning_rate"],
    max_depth=int(best_params["max_depth"]),
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

best_xgb_model.fit(
    X_train,
    y_train
)
# ============================================================
# 19. PREDICCIÓN FINAL VALIDATION
# ============================================================

y_pred_validation = best_xgb_model.predict(
    X_validation
)
# ============================================================
# 20. MÉTRICAS FINALES VALIDATION
# ============================================================

mae_val = mean_absolute_error(
    y_validation,
    y_pred_validation
)

rmse_val = np.sqrt(
    mean_squared_error(
        y_validation,
        y_pred_validation
    )
)

r2_val = r2_score(
    y_validation,
    y_pred_validation
)

mape_val = np.mean(
    np.abs(
        (y_validation - y_pred_validation)
        / y_validation
    )
) * 100

print("\nXGBOOST FINAL RESULTS")
print("MAE:", mae_val)
print("RMSE:", rmse_val)
print("R²:", r2_val)
print("MAPE:", mape_val)
# ============================================================
# 21. REAL VS PREDICHO
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

plt.scatter(
    y_validation,
    y_pred_validation,
    alpha=0.25
)

plt.plot(
    [
        y_validation.min(),
        y_validation.max()
    ],
    [
        y_validation.min(),
        y_validation.max()
    ],
    linestyle="--",
    linewidth=2
)

plt.title(
    "XGBoost Tuned - Real vs Predicho",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Valor real")
plt.ylabel("Valor predicho")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nTOP 15 FEATURES")
print(feature_importance.head(15))
# ============================================================
# 23. GRÁFICO FEATURE IMPORTANCE
# ============================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values("importance")
)

plt.figure(figsize=(10,7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "XGBoost Tuned - Top 15 Features",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Importancia")
plt.ylabel("Variable")

plt.tight_layout()
plt.show()

