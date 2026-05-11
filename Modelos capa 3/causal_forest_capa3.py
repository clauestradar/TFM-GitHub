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
# 7. CAUSAL FOREST - DATASET TRAIN
# ============================================================

df_cf_train = df_train.copy()

# ============================================================
# ELIMINAR NULOS EN VARIABLE DEPENDIENTE
# ============================================================

df_cf_train = df_cf_train.dropna(subset=["price_log"])

# ============================================================
# TREATMENT: RECESSION
# ============================================================

df_cf_train["recession_dummy"] = (
    df_cf_train["cycle_phase_new"] == "recession"
).astype(int)

print("\nDistribución treatment TRAIN:")
print(df_cf_train["recession_dummy"].value_counts())

# ============================================================
# OUTCOME
# ============================================================

Y_train = df_cf_train["price_log"].values

# ============================================================
# TREATMENT
# ============================================================

T_train = df_cf_train["recession_dummy"].values

# ============================================================
# VARIABLES HETEROGENEIDAD (X)
# ============================================================

X_train = pd.get_dummies(
    df_cf_train[[
        "segment_seed",
        "inflation_yoy",
        "unemployment_rate"
    ]],
    drop_first=True
)

# ============================================================
# CONTROLES (W)
# ============================================================

print("\nColumnas relacionadas con categoría:")
print([col for col in df_cf_train.columns if "cat" in col.lower()])

W_train = pd.get_dummies(
    df_cf_train[[
        "category_seed"
    ]],
    drop_first=True
)

print("W:", W_train.shape)

print("\n--- SHAPES TRAIN ---")
print("Y:", Y_train.shape)
print("T:", T_train.shape)
print("X:", X_train.shape)
print("W:", W_train.shape)
# ============================================================
# 8. ENTRENAMIENTO CAUSAL FOREST DML
# ============================================================

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

cf_model = CausalForestDML(
    model_y=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42
    ),
    model_t=RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42
    ),
    discrete_treatment=True,
    n_estimators=500,
    min_samples_leaf=30,
    max_depth=None,
    random_state=42
)

cf_model.fit(
    Y_train,
    T_train,
    X=X_train,
    W=W_train
)

print("\nCausal Forest entrenado correctamente.")
# ============================================================
# 9. ESTIMACIÓN DE EFECTOS CAUSALES HETEROGÉNEOS
# ============================================================

# Efecto causal individual estimado
tau_train = cf_model.effect(X_train)

df_effects_train = df_cf_train.copy()
df_effects_train["tau_recession"] = tau_train

print("\n--- RESUMEN EFECTO CAUSAL ESTIMADO: RECESIÓN ---")
print(df_effects_train["tau_recession"].describe())

print("\n--- EFECTO PROMEDIO POR SEGMENTO ---")
print(
    df_effects_train
    .groupby("segment_seed")["tau_recession"]
    .agg(["mean", "std", "count"])
)

print("\n--- EFECTO PROMEDIO POR CATEGORÍA ---")
print(
    df_effects_train
    .groupby("category_seed")["tau_recession"]
    .agg(["mean", "std", "count"])
)
# ============================================================
# 10. GRÁFICO EFECTO CAUSAL POR SEGMENTO
# ============================================================

import matplotlib.pyplot as plt

tau_segment = (
    df_effects_train
    .groupby("segment_seed")["tau_recession"]
    .mean()
    .reset_index()
    .sort_values("tau_recession")
)

plt.figure(figsize=(9, 5))

plt.bar(
    tau_segment["segment_seed"],
    tau_segment["tau_recession"],
    color=["#0d3b66", "#4ea8de", "#ff4fa3"]
)

plt.axhline(0, linestyle="--", linewidth=1.5, color="#0d3b66")

plt.title(
    "Efecto causal estimado de la recesión por segmento",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Segmento", fontsize=12)
plt.ylabel("Efecto estimado sobre price_log", fontsize=12)

plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
# ============================================================
# 11. CREAR CARPETA DE OUTPUTS
# ============================================================

import os

output_dir = "/Users/clau/Desktop/Máster Data Analytics /TFM/Código TFM/outputs/causal_forest"
os.makedirs(output_dir, exist_ok=True)
# ============================================================
# 12. HISTOGRAMA DE EFECTOS CAUSALES INDIVIDUALES
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df_effects_train["tau_recession"],
    bins=50,
    color="#4ea8de",
    edgecolor="#0d3b66",
    alpha=0.8
)

plt.axvline(
    df_effects_train["tau_recession"].mean(),
    color="#ff4fa3",
    linestyle="--",
    linewidth=2,
    label="Media CATE"
)

plt.axvline(
    0,
    color="#0d3b66",
    linestyle=":",
    linewidth=2,
    label="Efecto cero"
)

plt.title("Distribución de efectos causales estimados de la recesión", fontsize=15, fontweight="bold")
plt.xlabel("Efecto causal estimado sobre price_log")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(f"{output_dir}/histograma_cate_recession.png", dpi=300, bbox_inches="tight")
plt.show()
# ============================================================
# 13. BOXPLOT DE EFECTOS CAUSALES POR SEGMENTO
# ============================================================

plt.figure(figsize=(10, 6))

segment_order = ["mass", "mid", "premium"]

data_box = [
    df_effects_train[df_effects_train["segment_seed"] == seg]["tau_recession"]
    for seg in segment_order
]

box = plt.boxplot(
    data_box,
    labels=segment_order,
    patch_artist=True,
    showfliers=False
)

colors = ["#ff4fa3", "#4ea8de", "#0d3b66"]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

plt.axhline(0, color="#0d3b66", linestyle="--", linewidth=1.5)

plt.title("Distribución del efecto causal de la recesión por segmento", fontsize=15, fontweight="bold")
plt.xlabel("Segmento")
plt.ylabel("Efecto estimado sobre price_log")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(f"{output_dir}/boxplot_cate_segmento.png", dpi=300, bbox_inches="tight")
plt.show()
# ============================================================
# 14. EFECTO CAUSAL PROMEDIO POR CATEGORÍA
# ============================================================

tau_category = (
    df_effects_train
    .groupby("category_seed")["tau_recession"]
    .mean()
    .reset_index()
    .sort_values("tau_recession")
)

plt.figure(figsize=(10, 6))

plt.barh(
    tau_category["category_seed"],
    tau_category["tau_recession"],
    color="#4ea8de",
    edgecolor="#0d3b66"
)

plt.axvline(0, color="#ff4fa3", linestyle="--", linewidth=2)

plt.title("Efecto causal promedio de la recesión por categoría", fontsize=15, fontweight="bold")
plt.xlabel("Efecto estimado sobre price_log")
plt.ylabel("Categoría")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()

plt.savefig(f"{output_dir}/cate_promedio_categoria.png", dpi=300, bbox_inches="tight")
plt.show()
# ============================================================
# 15. GUARDAR RESULTADOS EN CSV
# ============================================================

df_effects_train.to_csv(
    f"{output_dir}/causal_forest_effects_train.csv",
    index=False
)

tau_segment.to_csv(
    f"{output_dir}/cate_promedio_segmento.csv",
    index=False
)

tau_category.to_csv(
    f"{output_dir}/cate_promedio_categoria.csv",
    index=False
)

print("\nResultados y gráficos guardados en:")
print(output_dir)