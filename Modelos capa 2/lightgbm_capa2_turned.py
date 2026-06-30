# ============================================================
# LIGHTGBM - CAPA 2 TFM
# Split temporal por meses completos
# Train / Test / Validation / Backtesting
# ============================================================
import pandas as pd
import numpy as np
import shap
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
    "segment_seed",
    "coverage_pct",
    "max_null_run",
    "low_quality_product",
    "was_imputed",
    "was_null",

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

# Guardar segmentos antes de eliminar segment_seed
segment_validation = df_validation["segment_seed"].copy()

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
# 16. GRID SEARCH LIGHTGBM
# ============================================================

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def calcular_mape(y_real, y_pred):
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100

results = []

learning_rates = [0.01, 0.05, 0.10]
max_depths = [3, 5, 7]
num_leaves_list = [31, 50]
subsamples = [0.8, 1.0]
colsamples = [0.8, 1.0]

for lr in learning_rates:
    for depth in max_depths:
        for leaves in num_leaves_list:
            for subs in subsamples:
                for cols in colsamples:

                    print(
                        f"\nEvaluando:"
                        f" lr={lr}"
                        f" depth={depth}"
                        f" leaves={leaves}"
                        f" subsample={subs}"
                        f" colsample={cols}"
                    )

                    model = LGBMRegressor(
                        n_estimators=500,
                        learning_rate=lr,
                        max_depth=depth,
                        num_leaves=leaves,
                        subsample=subs,
                        colsample_bytree=cols,
                        random_state=42,
                        verbose=-1
                    )

                    model.fit(X_train, y_train)

                    # Predicciones
                    y_pred_train = model.predict(X_train)
                    y_pred_validation = model.predict(X_validation)
                    y_pred_test = model.predict(X_test)

                    # Métricas train
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    mape_train = calcular_mape(y_train, y_pred_train)
                    r2_train = r2_score(y_train, y_pred_train)

                    # Métricas validation
                    mae_validation = mean_absolute_error(y_validation, y_pred_validation)
                    rmse_validation = np.sqrt(mean_squared_error(y_validation, y_pred_validation))
                    mape_validation = calcular_mape(y_validation, y_pred_validation)
                    r2_validation = r2_score(y_validation, y_pred_validation)

                    # Métricas test
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    mape_test = calcular_mape(y_test, y_pred_test)
                    r2_test = r2_score(y_test, y_pred_test)

                    results.append({
                        "learning_rate": lr,
                        "max_depth": depth,
                        "num_leaves": leaves,
                        "subsample": subs,
                        "colsample_bytree": cols,

                        "MAE_train": mae_train,
                        "RMSE_train": rmse_train,
                        "MAPE_train": mape_train,
                        "R2_train": r2_train,

                        "MAE_validation": mae_validation,
                        "RMSE_validation": rmse_validation,
                        "MAPE_validation": mape_validation,
                        "R2_validation": r2_validation,

                        "MAE_test": mae_test,
                        "RMSE_test": rmse_test,
                        "MAPE_test": mape_test,
                        "R2_test": r2_test
                    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["MAE_validation", "MAPE_validation"],
    ascending=[True, True]
)

print("\nRESULTADOS GRID SEARCH LIGHTGBM")
print(results_df.head(10))

# Guardar tabla de hiperparámetros
results_df.to_csv(
    "lightgbm_grid_search_results.csv",
    index=False
)

print("\nTabla de hiperparámetros guardada como:")
print("lightgbm_grid_search_results.csv")
# ============================================================
# 17. SELECCIÓN MODELO LIGHTGBM RECOMENDADO POR ASESOR
# Modelo recomendado: learning_rate=0.01, max_depth=7,
# num_leaves=50, subsample=0.8, colsample_bytree=0.8
# ============================================================

best_params = results_df[
    (results_df["learning_rate"] == 0.01) &
    (results_df["max_depth"] == 7) &
    (results_df["num_leaves"] == 50) &
    (results_df["subsample"] == 0.8) &
    (results_df["colsample_bytree"] == 0.8)
].iloc[0]

print("\nMODELO LIGHTGBM SELECCIONADO POR ASESOR")
print(best_params)
# ============================================================
# 18. ENTRENAR MEJOR LIGHTGBM
# ============================================================

best_lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=best_params["learning_rate"],
    max_depth=int(best_params["max_depth"]),
    num_leaves=int(best_params["num_leaves"]),
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
    verbose=-1
)

best_lgbm_model.fit(
    X_train,
    y_train
)

print("\nMejor modelo LightGBM entrenado correctamente.")
# ============================================================
# 19. PREDICCIONES FINALES TRAIN / TEST / VALIDATION
# ============================================================

y_pred_train_lgbm = best_lgbm_model.predict(X_train)
y_pred_test_lgbm = best_lgbm_model.predict(X_test)
y_pred_validation_lgbm = best_lgbm_model.predict(X_validation)

# ============================================================
# 20. MÉTRICAS FINALES TRAIN / TEST / VALIDATION
# ============================================================

def calcular_metricas_finales(y_real, y_pred):
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE": calcular_mape(y_real, y_pred),
        "R2": r2_score(y_real, y_pred)
    }

metrics_lgbm = {
    "Train": calcular_metricas_finales(y_train, y_pred_train_lgbm),
    "Test": calcular_metricas_finales(y_test, y_pred_test_lgbm),
    "Validation": calcular_metricas_finales(y_validation, y_pred_validation_lgbm)
}

metrics_lgbm_df = (
    pd.DataFrame(metrics_lgbm)
    .T
    .reset_index()
    .rename(columns={"index": "fase"})
)

print("\nLIGHTGBM FINAL RESULTS - TRAIN / TEST / VALIDATION")
print(metrics_lgbm_df)

metrics_lgbm_df.to_csv(
    "lightgbm_final_train_test_validation_metrics.csv",
    index=False
)

print("\nMétricas finales guardadas como:")
print("lightgbm_final_train_test_validation_metrics.csv")
# ============================================================
# 21. REAL VS PREDICHO
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

plt.scatter(
    y_validation,
    y_pred_validation_lgbm,
    alpha=0.25
)

plt.plot(
    [y_validation.min(), y_validation.max()],
    [y_validation.min(), y_validation.max()],
    linestyle="--",
    linewidth=2
)

plt.title(
    "LightGBM Tuned - Real vs Predicho",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Valor real")
plt.ylabel("Valor predicho")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# ============================================================
# 22. FEATURE IMPORTANCE DEL MEJOR MODELO LIGHTGBM (%)
# ============================================================

feature_importance_lgbm = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_lgbm_model.feature_importances_
})

feature_importance_lgbm["importance_pct"] = (
    feature_importance_lgbm["importance"]
    / feature_importance_lgbm["importance"].sum()
) * 100

feature_importance_lgbm = feature_importance_lgbm.sort_values(
    "importance_pct",
    ascending=False
)

print("\nTOP 15 FEATURES LIGHTGBM - IMPORTANCIA (%)")
print(
    feature_importance_lgbm[
        ["feature", "importance_pct"]
    ].head(15)
)

feature_importance_lgbm.to_csv(
    "lightgbm_feature_importance_pct.csv",
    index=False
)

# ============================================================
# 23. GRÁFICO FEATURE IMPORTANCE DEL MEJOR MODELO (%)
# ============================================================

top_features_lgbm = (
    feature_importance_lgbm
    .head(15)
    .sort_values("importance_pct")
)

plt.figure(figsize=(10,7))

plt.barh(
    top_features_lgbm["feature"],
    top_features_lgbm["importance_pct"]
)

plt.title(
    "LightGBM Tuned - Top 15 Variables más importantes (%)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Importancia relativa (%)")
plt.ylabel("Variable")

plt.tight_layout()
plt.show()
# ============================================================
# 24. SHAP VALUES
# ============================================================

import shap

print("\nCalculando SHAP values...")

# Crear explainer
explainer = shap.TreeExplainer(best_lgbm_model)

# Calcular SHAP values
shap_values = explainer.shap_values(X_validation)

print("SHAP values calculados correctamente.")
# ============================================================
# 25. SHAP SUMMARY PLOT GLOBAL
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_validation,
    show=False
)

plt.title(
    "SHAP Summary Plot - LightGBM",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "shap_summary_global.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# 26. SHAP BEESWARM POR SEGMENTO
# ============================================================

segments = ["mass", "mid", "premium"]

for seg in segments:

    print(f"\nGenerando SHAP beeswarm para: {seg}")

    mask = segment_validation.loc[X_validation.index] == seg

    X_seg = X_validation[mask]

    shap_seg = explainer.shap_values(X_seg)

    plt.figure()

    shap.summary_plot(
        shap_seg,
        X_seg,
        show=False
    )

    plt.title(
        f"SHAP Beeswarm - Segmento {seg}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"shap_beeswarm_{seg}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    # ============================================================
# 27. SHAP DEPENDENCE PLOTS
# ============================================================

variables_dependence = [
    "cycle_phase",
    "cpi_beauty"
]

for var in variables_dependence:

    if var in X_validation.columns:

        plt.figure()

        shap.dependence_plot(
            var,
            shap_values,
            X_validation,
            show=False
        )

        plt.title(
            f"SHAP Dependence Plot - {var}",
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            f"shap_dependence_{var}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        # ============================================================
# 28. PARTIAL DEPENDENCE PLOTS (PDP)
# Variables macroeconómicas: cpi_beauty y cycle_phase
# ============================================================

from sklearn.inspection import PartialDependenceDisplay

print("\nGenerando Partial Dependence Plots (PDP)...")

pdp_features = [
    var for var in ["cpi_beauty", "cycle_phase"]
    if var in X_train.columns
]

if len(pdp_features) == 0:
    print("No se encontraron variables disponibles para PDP.")
else:

    # PDP conjunto, como pidió el asesor
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(pdp_features),
        figsize=(6 * len(pdp_features), 5)
    )

    PartialDependenceDisplay.from_estimator(
        best_lgbm_model,
        X_train,
        features=pdp_features,
        ax=ax
    )

    plt.suptitle(
        "Partial Dependence Plots - LightGBM",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        "pdp_lightgbm_cpi_beauty_cycle_phase.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("PDP conjunto guardado como:")
    print("pdp_lightgbm_cpi_beauty_cycle_phase.png")

    # PDP individuales para que queden más claros en la memoria
    for var in pdp_features:

        print(f"\nGenerando PDP individual para: {var}")

        fig, ax = plt.subplots(figsize=(8, 6))

        display = PartialDependenceDisplay.from_estimator(
            best_lgbm_model,
            X_train,
            features=[var],
            ax=ax
        )

        ax.set_title(
            f"Partial Dependence Plot - {var}",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel(var)
        ax.set_ylabel("Predicción promedio")
        ax.grid(alpha=0.3)

        # Si cycle_phase fue codificada con LabelEncoder, recuperar etiquetas originales
        if var == "cycle_phase" and "cycle_phase" in encoders:
            labels = list(encoders["cycle_phase"].classes_)
            ticks = range(len(labels))
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()

        filename = f"pdp_lightgbm_{var}.png"

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print("PDP individual guardado como:")
        print(filename)

print("\nBloque PDP finalizado correctamente.")
# ============================================================
# BACKTESTING FINAL
# ============================================================

y_pred_backtest_lgbm = best_lgbm_model.predict(X_backtest)

mae_backtest = mean_absolute_error(
    y_backtest,
    y_pred_backtest_lgbm
)

rmse_backtest = np.sqrt(
    mean_squared_error(
        y_backtest,
        y_pred_backtest_lgbm
    )
)

mape_backtest = calcular_mape(
    y_backtest,
    y_pred_backtest_lgbm
)

r2_backtest = r2_score(
    y_backtest,
    y_pred_backtest_lgbm
)

accuracy_backtest = 100 - mape_backtest

print("\nRESULTADOS BACKTESTING")
print(f"MAE: {mae_backtest:.4f}")
print(f"RMSE: {rmse_backtest:.4f}")
print(f"MAPE: {mape_backtest:.4f}")
print(f"R2: {r2_backtest:.4f}")
print(f"Accuracy Forecast: {accuracy_backtest:.2f}%")
backtest_df = pd.DataFrame({
    "Fase": ["Backtesting"],
    "MAE": [mae_backtest],
    "RMSE": [rmse_backtest],
    "MAPE": [mape_backtest],
    "R2": [r2_backtest],
    "Accuracy_Forecast": [accuracy_backtest]
})

backtest_df.to_csv(
    "lightgbm_backtesting_metrics.csv",
    index=False
)