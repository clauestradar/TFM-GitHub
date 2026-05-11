# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# 2. CARGA DEL DATASET
# ============================================================

ruta = "/Users/clau/Desktop/Máster Data Analytics /TFM/Código TFM/analytic_panel_final_FINAL.csv"

df = pd.read_csv(ruta)

df["month"] = pd.to_datetime(df["month"])

print("Shape dataset completo:", df.shape)
print("Fecha mínima:", df["month"].min())
print("Fecha máxima:", df["month"].max())

# ============================================================
# 3. SPLIT TEMPORAL POR MESES COMPLETOS
# ============================================================

meses = sorted(df["month"].unique())
n_meses = len(meses)

print("\nNúmero total de meses:", n_meses)

# 10% final para backtesting
n_backtesting = int(np.ceil(n_meses * 0.10))

meses_backtesting = meses[-n_backtesting:]
meses_modelado = meses[:-n_backtesting]

df_modelado = df[df["month"].isin(meses_modelado)].copy()
df_backtesting = df[df["month"].isin(meses_backtesting)].copy()

print("\n--- BACKTESTING HOLDOUT 10% ---")
print("Shape:", df_backtesting.shape)
print("Fechas:", df_backtesting["month"].min(), "→", df_backtesting["month"].max())
print("Meses:", df_backtesting["month"].nunique())

print("\n--- DATASET MODELADO 90% ---")
print("Shape:", df_modelado.shape)
print("Fechas:", df_modelado["month"].min(), "→", df_modelado["month"].max())
print("Meses:", df_modelado["month"].nunique())
# ============================================================
# 4. SPLIT VALIDACIÓN DENTRO DEL 90%
# ============================================================

meses_modelado_ordenados = sorted(df_modelado["month"].unique())
n_meses_modelado = len(meses_modelado_ordenados)

n_validacion = int(np.ceil(n_meses_modelado * 0.30))

meses_validacion = meses_modelado_ordenados[-n_validacion:]
meses_train_test = meses_modelado_ordenados[:-n_validacion]

df_validacion = df_modelado[df_modelado["month"].isin(meses_validacion)].copy()
df_train_test = df_modelado[df_modelado["month"].isin(meses_train_test)].copy()

print("\n--- VALIDACIÓN 30% DEL DATASET DE MODELADO ---")
print("Shape:", df_validacion.shape)
print("Fechas:", df_validacion["month"].min(), "→", df_validacion["month"].max())
print("Meses:", df_validacion["month"].nunique())

print("\n--- BLOQUE TRAIN/TEST 70% DEL DATASET DE MODELADO ---")
print("Shape:", df_train_test.shape)
print("Fechas:", df_train_test["month"].min(), "→", df_train_test["month"].max())
print("Meses:", df_train_test["month"].nunique())
# ============================================================
# 5. SPLIT TRAIN / TEST DENTRO DEL BLOQUE TRAIN_TEST
# ============================================================

meses_train_test_ordenados = sorted(df_train_test["month"].unique())
n_meses_train_test = len(meses_train_test_ordenados)

n_test = int(np.ceil(n_meses_train_test * 0.30))

meses_test = meses_train_test_ordenados[-n_test:]
meses_train = meses_train_test_ordenados[:-n_test]

df_train = df_train_test[df_train_test["month"].isin(meses_train)].copy()
df_test = df_train_test[df_train_test["month"].isin(meses_test)].copy()

print("\n--- TRAIN ---")
print("Shape:", df_train.shape)
print("Fechas:", df_train["month"].min(), "→", df_train["month"].max())
print("Meses:", df_train["month"].nunique())

print("\n--- TEST ---")
print("Shape:", df_test.shape)
print("Fechas:", df_test["month"].min(), "→", df_test["month"].max())
print("Meses:", df_test["month"].nunique())

# ============================================================
# 6. CONTROL FINAL
# ============================================================

print("\n--- CONTROL FINAL DE SPLITS ---")
print("Meses totales:", df["month"].nunique())
print("Train:", df_train["month"].nunique())
print("Test:", df_test["month"].nunique())
print("Validación:", df_validacion["month"].nunique())
print("Backtesting:", df_backtesting["month"].nunique())

print("\nFilas totales:", len(df))
print("Filas asignadas:", len(df_train) + len(df_test) + len(df_validacion) + len(df_backtesting))
# ============================================================
# 7. PREPARACIÓN DEL PANEL PARA EFECTOS FIJOS
# ============================================================

df_fe_train = df_train.copy()

# Variable dependiente
y_col = "price_log"

print(f"\nResumen {y_col}:")
print(df_fe_train[y_col].describe())

print("\nASINs únicos en train:", df_fe_train["asin"].nunique())
print("Meses únicos en train:", df_fe_train["month"].nunique())
print("Filas train:", df_fe_train.shape[0])
# ============================================================
# 8. DATASET TRAIN LIMPIO PARA PANELOLS
# ============================================================

df_fe_train_model = df_fe_train.copy()

# Nos quedamos solo con filas donde la variable dependiente existe
df_fe_train_model = df_fe_train_model.dropna(subset=[y_col])

print("\n--- TRAIN LIMPIO PARA MODELO ---")
print("Shape:", df_fe_train_model.shape)
print("ASINs únicos:", df_fe_train_model["asin"].nunique())
print("Meses únicos:", df_fe_train_model["month"].nunique())
print("Nulos en variable dependiente:", df_fe_train_model[y_col].isna().sum())
# ============================================================
# 9. VARIABLES DISPONIBLES PARA EL MODELO
# ============================================================

vars_modelo = [
    "inflation_yoy",
    "unemployment_rate",
    "interest_rate",
    "real_interest_rate",
    "post_pandemic",
    "high_rate_regime",
    "cycle_phase_new"
]

print("\n--- DISPONIBILIDAD VARIABLES ---")

for col in vars_modelo:
    print(col, "→", col in df_fe_train_model.columns)
# ============================================================
# 10. PANELOLS - EFECTOS FIJOS POR PRODUCTO
# ============================================================

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

X_vars = [
    "inflation_yoy",
    "unemployment_rate",
    "interest_rate"
]

df_panel = df_fe_train_model.copy()

df_panel = df_panel[
    ["asin", "month", y_col] + X_vars
].dropna()

print("\n--- DATASET FINAL PANEL ---")
print(df_panel.shape)

df_panel = df_panel.set_index(["asin", "month"])

y = df_panel[y_col]
X = df_panel[X_vars]

X = sm.add_constant(X)

modelo_fe = PanelOLS(
    y,
    X,
    entity_effects=True
)

resultado_fe = modelo_fe.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(resultado_fe.summary)
print("\nSegmentos disponibles:")
print(df_fe_train_model["segment_seed"].value_counts())
print("\nFases ciclo disponibles:")
print(df_fe_train_model["cycle_phase_new"].value_counts())
# ============================================================
# 11. MODELO EFECTOS FIJOS:
# CICLO ECONÓMICO + INTERACCIONES POR SEGMENTO
# ============================================================

# Copia dataset
df_inter = df_fe_train_model.copy()

# ============================================================
# DUMMIES DE CICLO
# ============================================================

cycle_dummies = pd.get_dummies(
    df_inter["cycle_phase_new"],
    prefix="cycle",
    drop_first=True
)

df_inter = pd.concat([df_inter, cycle_dummies], axis=1)

# ============================================================
# INTERACCIONES PREMIUM × CICLO
# ============================================================

df_inter["premium_dummy"] = (
    df_inter["segment_seed"] == "premium"
).astype(int)

for col in cycle_dummies.columns:
    df_inter[f"{col}_x_premium"] = (
        df_inter[col] * df_inter["premium_dummy"]
    )

# ============================================================
# VARIABLES MODELO
# ============================================================

X_vars_inter = (
    list(cycle_dummies.columns)
    + [f"{col}_x_premium" for col in cycle_dummies.columns]
)

print("\nVariables modelo interacción:")
print(X_vars_inter)

# ============================================================
# DATASET FINAL
# ============================================================

df_panel_inter = df_inter[
    ["asin", "month", y_col] + X_vars_inter
].dropna()

print("\n--- DATASET INTERACCIONES ---")
print(df_panel_inter.shape)

# ============================================================
# PANEL INDEX
# ============================================================

df_panel_inter = df_panel_inter.set_index(["asin", "month"])

y_inter = df_panel_inter[y_col]

X_inter = df_panel_inter[X_vars_inter]

X_inter = sm.add_constant(X_inter)

# ============================================================
# MODELO
# ============================================================

modelo_inter = PanelOLS(
    y_inter,
    X_inter,
    entity_effects=True
)

resultado_inter = modelo_inter.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(resultado_inter.summary)
# ============================================================
# 12. MODELO FINAL:
# INTERACCIONES POR SEGMENTO COMPLETO
# ============================================================

df_seg = df_fe_train_model.copy()

# ============================================================
# DUMMIES CICLO
# ============================================================

cycle_dummies = pd.get_dummies(
    df_seg["cycle_phase_new"],
    prefix="cycle",
    drop_first=True
)

df_seg = pd.concat([df_seg, cycle_dummies], axis=1)

# ============================================================
# DUMMIES SEGMENTO
# MASS = baseline
# ============================================================

df_seg["mid_dummy"] = (
    df_seg["segment_seed"] == "mid"
).astype(int)

df_seg["premium_dummy"] = (
    df_seg["segment_seed"] == "premium"
).astype(int)

# ============================================================
# INTERACCIONES MID
# ============================================================

for col in cycle_dummies.columns:
    df_seg[f"{col}_x_mid"] = (
        df_seg[col] * df_seg["mid_dummy"]
    )

# ============================================================
# INTERACCIONES PREMIUM
# ============================================================

for col in cycle_dummies.columns:
    df_seg[f"{col}_x_premium"] = (
        df_seg[col] * df_seg["premium_dummy"]
    )

# ============================================================
# VARIABLES
# ============================================================

X_vars_seg = (
    list(cycle_dummies.columns)
    + [f"{col}_x_mid" for col in cycle_dummies.columns]
    + [f"{col}_x_premium" for col in cycle_dummies.columns]
)

print("\nVariables modelo final:")
print(X_vars_seg)

# ============================================================
# DATASET FINAL
# ============================================================

df_panel_seg = df_seg[
    ["asin", "month", y_col] + X_vars_seg
].dropna()

print("\n--- DATASET MODELO FINAL ---")
print(df_panel_seg.shape)

# ============================================================
# PANEL INDEX
# ============================================================

df_panel_seg = df_panel_seg.set_index(["asin", "month"])

y_seg = df_panel_seg[y_col]

X_seg = df_panel_seg[X_vars_seg]

X_seg = sm.add_constant(X_seg)

# ============================================================
# MODELO
# ============================================================

modelo_seg = PanelOLS(
    y_seg,
    X_seg,
    entity_effects=True
)

resultado_seg = modelo_seg.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(resultado_seg.summary)
# ============================================================
# 13. GRÁFICO DE COEFICIENTES - MODELO FINAL
# ============================================================

import matplotlib.pyplot as plt

# Extraer coeficientes e intervalos de confianza
coef = resultado_seg.params
conf = resultado_seg.conf_int()
pvalues = resultado_seg.pvalues

coef_df = pd.DataFrame({
    "variable": coef.index,
    "coeficiente": coef.values,
    "ci_lower": conf.iloc[:, 0].values,
    "ci_upper": conf.iloc[:, 1].values,
    "p_value": pvalues.values
})

# Quitamos la constante para graficar
coef_df = coef_df[coef_df["variable"] != "const"].copy()

# Ordenamos por coeficiente
coef_df = coef_df.sort_values("coeficiente")

print("\n--- COEFICIENTES MODELO FINAL ---")
print(coef_df)

plt.figure(figsize=(11, 6))

plt.errorbar(
    coef_df["coeficiente"],
    coef_df["variable"],
    xerr=[
        coef_df["coeficiente"] - coef_df["ci_lower"],
        coef_df["ci_upper"] - coef_df["coeficiente"]
    ],
    fmt="o",
    color="#ff4fa3",
    ecolor="#0d3b66",
    elinewidth=2,
    capsize=4,
    markersize=8
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
    color="#0d3b66"
)

plt.title(
    "Coeficientes del modelo PanelOLS con efectos fijos",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Coeficiente estimado sobre price_log",
    fontsize=12
)

plt.ylabel(
    "Variable",
    fontsize=12
)

plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()
# ============================================================
# 14. VALIDACIÓN TEMPORAL DEL MODELO FINAL
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_val = df_validacion.copy()

# Nos quedamos con filas donde existe la variable dependiente
df_val = df_val.dropna(subset=[y_col])

# Creamos las mismas dummies de ciclo
cycle_dummies_val = pd.get_dummies(
    df_val["cycle_phase_new"],
    prefix="cycle",
    drop_first=True
)

df_val = pd.concat([df_val, cycle_dummies_val], axis=1)

# Asegurar que tenga las mismas columnas de ciclo que train
for col in cycle_dummies.columns:
    if col not in df_val.columns:
        df_val[col] = 0

# Dummies segmento
df_val["mid_dummy"] = (
    df_val["segment_seed"] == "mid"
).astype(int)

df_val["premium_dummy"] = (
    df_val["segment_seed"] == "premium"
).astype(int)

# Interacciones MID y PREMIUM
for col in cycle_dummies.columns:
    df_val[f"{col}_x_mid"] = df_val[col] * df_val["mid_dummy"]
    df_val[f"{col}_x_premium"] = df_val[col] * df_val["premium_dummy"]

# Dataset validación
df_panel_val = df_val[
    ["asin", "month", y_col] + X_vars_seg
].dropna()

print("\n--- DATASET VALIDACIÓN MODELO FINAL ---")
print(df_panel_val.shape)

# Índice panel
df_panel_val = df_panel_val.set_index(["asin", "month"])

y_val = df_panel_val[y_col]
X_val = df_panel_val[X_vars_seg]

X_val = sm.add_constant(X_val, has_constant="add")

# Asegurar mismo orden de columnas que el modelo entrenado
X_val = X_val[resultado_seg.params.index]

# Predicción
pred_val = resultado_seg.predict(X_val)

# Convertir a Series limpia
y_pred_val = pred_val.iloc[:, 0]

# Métricas en escala log
mae_log = mean_absolute_error(y_val, y_pred_val)
rmse_log = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)

print("\n--- MÉTRICAS VALIDACIÓN EN ESCALA LOG ---")
print("MAE log:", mae_log)
print("RMSE log:", rmse_log)
print("R² validación:", r2_val)
# ============================================================
# 15. PREPARAR DATAFRAME PARA GRÁFICO VALIDACIÓN
# ============================================================

df_val_plot = pd.DataFrame({
    "real": y_val,
    "predicho": y_pred_val
}).reset_index()

df_val_monthly = (
    df_val_plot
    .groupby("month")[["real", "predicho"]]
    .mean()
    .reset_index()
)

print("\n--- VALIDACIÓN PROMEDIO MENSUAL ---")
print(df_val_monthly.head())
plt.figure(figsize=(12, 6))

plt.plot(
    df_val_monthly["month"],
    df_val_monthly["real"],
    marker="o",
    linewidth=2.5,
    color="#0d3b66",
    label="Real"
)

plt.plot(
    df_val_monthly["month"],
    df_val_monthly["predicho"],
    marker="o",
    linewidth=2.5,
    color="#ff4fa3",
    label="Predicho"
)

plt.title(
    "Validación temporal: price_log real vs predicho",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Mes",
    fontsize=12
)

plt.ylabel(
    "price_log promedio",
    fontsize=12
)

plt.grid(alpha=0.2)

plt.legend()

plt.tight_layout()
plt.show()
# ============================================================
# 16. GRÁFICO REAL VS PREDICHO - VALIDACIÓN
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df_val_monthly["month"],
    df_val_monthly["real"],
    marker="o",
    linewidth=2.5,
    color="#0d3b66",
    label="Real"
)

plt.plot(
    df_val_monthly["month"],
    df_val_monthly["predicho"],
    marker="o",
    linewidth=2.5,
    color="#ff4fa3",
    label="Predicho"
)

plt.title(
    "Validación temporal: price_log real vs predicho",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Mes", fontsize=12)
plt.ylabel("price_log promedio", fontsize=12)

plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()