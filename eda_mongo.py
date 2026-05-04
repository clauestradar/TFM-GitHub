# %%
# ============================================================
# EDA TFM COSMETICS — DESDE MONGODB
# Autor: Clau
# Proyecto: Lipstick Index / Cosmetic Pricing
# ============================================================


# %%
# ============================================================
# 0. IMPORTS Y ESTILO VISUAL
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymongo import MongoClient

sns.set_theme(style="whitegrid")

PALETTE_BLUE_PINK = [
    "#0d3b66",  # azul oscuro
    "#1f77b4",  # azul medio
    "#4ea8de",  # azul claro
    "#ff4fa3",  # rosado fuerte
    "#f7a1c4",  # rosado suave
    "#b5179e"   # magenta
]

sns.set_palette(PALETTE_BLUE_PINK)

plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12


# %%
# ============================================================
# 1. CONEXIÓN A MONGODB
# ============================================================

client = MongoClient("mongodb://localhost:27017/")

db = client["tfm_cosmetics"]
collection = db["analytic_panel_final"]

print("Colecciones disponibles:")
print(db.list_collection_names())

print("\nNúmero de documentos:")
print(collection.count_documents({}))


# %%
# ============================================================
# 2. CARGA DE DATOS DESDE MONGO A DATAFRAME
# ============================================================

data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

print("Shape del DataFrame:")
print(df.shape)

print("\nPrimeras filas:")
print(df.head())

print("\nColumnas disponibles:")
print(df.columns.tolist())


# %%
# ============================================================
# 3. PREPARACIÓN BÁSICA DEL PANEL
# ============================================================

df["month"] = pd.to_datetime(df["month"], errors="coerce")

df = df.sort_values(["asin", "month"]).reset_index(drop=True)

print("Rango temporal:")
print(df["month"].min(), "→", df["month"].max())

print("\nNúmero de ASINs únicos:")
print(df["asin"].nunique())

print("\nMeses por ASIN:")
print(df.groupby("asin")["month"].nunique().describe())


# %%
# ============================================================
# 4. DEFINICIÓN DEL TARGET
# ============================================================

# Variable objetivo principal: precio real observado en Amazon
df["target_price"] = df["amazon_price"]

# Variable logarítmica del precio
df["log_target_price"] = np.where(
    df["target_price"] > 0,
    np.log(df["target_price"]),
    np.nan
)

print("Missing target_price:")
print(df["target_price"].isna().mean())

print("\nMissing log_target_price:")
print(df["log_target_price"].isna().mean())

print("\nResumen target_price:")
print(df["target_price"].describe())

print("\nResumen log_target_price:")
print(df["log_target_price"].describe())


# %%
# ============================================================
# 5. VALIDACIÓN DE SEGMENTOS Y CATEGORÍAS
# ============================================================

print("Distribución por segmento:")
print(df["segment_seed"].value_counts(dropna=False))

print("\nDistribución por categoría:")
print(df["category_seed"].value_counts(dropna=False))

print("\nASINs únicos por segmento:")
print(df.groupby("segment_seed")["asin"].nunique())

print("\nASINs únicos por categoría:")
print(df.groupby("category_seed")["asin"].nunique())

print("\nCruce categoría x segmento:")
print(pd.crosstab(df["category_seed"], df["segment_seed"]))


# %%
# ============================================================
# 6. DATAFRAME FILTRADO PARA EDA DE PRECIOS REALES
# ============================================================

df_prices = df[
    df["target_price"].notna() &
    (df["target_price"] > 0)
].copy()

print("Observaciones totales:", df.shape[0])
print("Observaciones con precio válido:", df_prices.shape[0])
print("Porcentaje útil:", round(df_prices.shape[0] / df.shape[0] * 100, 2), "%")


# %%
# ============================================================
# 7. DISTRIBUCIÓN GLOBAL DE PRECIOS
# ============================================================

plt.figure()
sns.histplot(
    df_prices["target_price"],
    bins=50,
    kde=True,
    color="#1f77b4"
)
plt.title("Distribución global de precios")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 8. DISTRIBUCIÓN DE PRECIOS POR SEGMENTO
# ============================================================

plt.figure()
sns.histplot(
    data=df_prices,
    x="target_price",
    hue="segment_seed",
    bins=50,
    kde=True,
    element="step",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Distribución de precios por segmento")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 9. BOXPLOT DE PRECIOS POR SEGMENTO
# ============================================================

plt.figure()
sns.boxplot(
    data=df_prices,
    x="segment_seed",
    y="target_price",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Boxplot de precios por segmento")
plt.xlabel("Segmento")
plt.ylabel("Precio")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 10. BOXPLOT DE PRECIOS POR CATEGORÍA
# ============================================================

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_prices,
    x="category_seed",
    y="target_price",
    palette=PALETTE_BLUE_PINK
)
plt.title("Boxplot de precios por categoría")
plt.xlabel("Categoría")
plt.ylabel("Precio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# ============================================================

# 11. VARIABLES MACRO DISPONIBLES
# ============================================================

macro_candidates = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in [
        "fred", "rate", "cpi", "inflation", "unemployment",
        "gdp", "interest", "fed", "macro", "cycle"
    ])
]

print("Variables macro / ciclo candidatas:")
print(macro_candidates)


# %%
# ============================================================
# 12. EVOLUCIÓN TEMPORAL DEL PRECIO PROMEDIO GLOBAL
# ============================================================

price_time = (
    df_prices
    .groupby("month")["target_price"]
    .mean()
    .reset_index()
)

plt.figure()
sns.lineplot(
    data=price_time,
    x="month",
    y="target_price",
    color="#1f77b4",
    linewidth=2.5
)
plt.title("Evolución temporal del precio promedio")
plt.xlabel("Fecha")
plt.ylabel("Precio promedio")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 13. EVOLUCIÓN TEMPORAL DEL PRECIO POR SEGMENTO
# ============================================================

price_segment = (
    df_prices
    .groupby(["month", "segment_seed"])["target_price"]
    .mean()
    .reset_index()
)

plt.figure()
sns.lineplot(
    data=price_segment,
    x="month",
    y="target_price",
    hue="segment_seed",
    linewidth=2.5,
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Evolución temporal del precio promedio por segmento")
plt.xlabel("Fecha")
plt.ylabel("Precio promedio")
plt.legend(title="Segmento")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 14. DISTRIBUCIÓN DE PRECIOS POR FASE DEL CICLO ECONÓMICO
# ============================================================

plt.figure()
sns.boxplot(
    data=df_prices,
    x="cycle_phase_new",
    y="target_price",
    palette=PALETTE_BLUE_PINK
)
plt.title("Distribución de precios por fase del ciclo económico")
plt.xlabel("Fase del ciclo económico")
plt.ylabel("Precio")
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 15. PRECIO PROMEDIO POR FASE DEL CICLO Y SEGMENTO
# ============================================================

cycle_segment_summary = (
    df_prices
    .groupby(["cycle_phase_new", "segment_seed"])["target_price"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

print(cycle_segment_summary)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=cycle_segment_summary,
    x="cycle_phase_new",
    y="mean",
    hue="segment_seed",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Precio promedio por fase del ciclo económico y segmento")
plt.xlabel("Fase del ciclo económico")
plt.ylabel("Precio promedio")
plt.legend(title="Segmento")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 16. DATASET MENSUAL PARA OVERLAYS MACRO
# ============================================================

macro_monthly = (
    df_prices
    .groupby("month")
    .agg({
        "target_price": "mean",
        "cpi": "mean",
        "cpi_beauty": "mean",
        "unemployment_rate": "mean",
        "interest_rate": "mean",
        "inflation_yoy": "mean",
        "real_interest_rate": "mean"
    })
    .reset_index()
)

print(macro_monthly.head())

# %%
# ============================================================
# 17. PRECIO VS VARIABLES MACRO — OVERLAYS
# ============================================================

macro_vars = [
    "cpi",
    "cpi_beauty",
    "inflation_yoy",
    "interest_rate",
    "real_interest_rate",
    "unemployment_rate"
]

for var in macro_vars:
    fig, ax1 = plt.subplots(figsize=(11, 6))

    sns.lineplot(
        data=macro_monthly,
        x="month",
        y="target_price",
        ax=ax1,
        color="#0d3b66",
        linewidth=2.5,
        label="Precio promedio"
    )

    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precio promedio")

    ax2 = ax1.twinx()

    sns.lineplot(
        data=macro_monthly,
        x="month",
        y=var,
        ax=ax2,
        color="#ff4fa3",
        linewidth=2.2,
        linestyle="--",
        label=var
    )

    ax2.set_ylabel(var)

    plt.title(f"Evolución del precio promedio vs {var}")
    fig.tight_layout()
    plt.show()

    # %%
# ============================================================
# 18. CORRELACIÓN ENTRE PRECIO Y VARIABLES MACRO
# ============================================================

corr_vars = [
    "target_price",
    "cpi",
    "cpi_beauty",
    "inflation_yoy",
    "interest_rate",
    "real_interest_rate",
    "unemployment_rate"
]

corr_matrix = macro_monthly[corr_vars].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdPu",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Matriz de correlación: precio promedio y variables macro")
plt.tight_layout()
plt.show()

print(corr_matrix["target_price"].sort_values(ascending=False))

# %%
# ============================================================
# 19. IMPORTS PARA ESTACIONARIEDAD Y SERIES TEMPORALES
# ============================================================

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.filterwarnings("ignore")

# %%
# ============================================================
# 20. SERIE MENSUAL GLOBAL DE PRECIO PROMEDIO
# ============================================================

price_monthly = (
    df_prices
    .groupby("month")["target_price"]
    .mean()
    .asfreq("MS")
)

price_monthly = price_monthly.interpolate(method="linear")

print(price_monthly.head())
print(price_monthly.tail())

plt.figure(figsize=(11, 6))
plt.plot(price_monthly, color="#0d3b66", linewidth=2.5)
plt.title("Serie mensual del precio promedio")
plt.xlabel("Fecha")
plt.ylabel("Precio promedio")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 21. FUNCIONES ADF Y KPSS
# ============================================================

def adf_test(series, name="serie"):
    series = series.dropna()
    result = adfuller(series, autolag="AIC")

    return {
        "serie": name,
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "lags": result[2],
        "n_obs": result[3],
        "conclusion": "Estacionaria" if result[1] < 0.05 else "No estacionaria"
    }


def kpss_test(series, name="serie"):
    series = series.dropna()
    result = kpss(series, regression="c", nlags="auto")

    return {
        "serie": name,
        "test": "KPSS",
        "statistic": result[0],
        "p_value": result[1],
        "lags": result[2],
        "n_obs": len(series),
        "conclusion": "No estacionaria" if result[1] < 0.05 else "Estacionaria"
    }

# %%
# ============================================================
# 22. ADF Y KPSS — PRECIO PROMEDIO GLOBAL
# ============================================================

stationarity_results = []

stationarity_results.append(adf_test(price_monthly, "precio_promedio_global"))
stationarity_results.append(kpss_test(price_monthly, "precio_promedio_global"))

stationarity_df = pd.DataFrame(stationarity_results)

print(stationarity_df)

# %%
# ============================================================
# 23. DIFERENCIACIÓN DE LA SERIE GLOBAL
# ============================================================

price_monthly_diff = price_monthly.diff().dropna()

plt.figure(figsize=(11, 6))
plt.plot(price_monthly_diff, color="#ff4fa3", linewidth=2.5)
plt.title("Primera diferencia del precio promedio")
plt.xlabel("Fecha")
plt.ylabel("Diferencia mensual del precio promedio")
plt.tight_layout()
plt.show()

diff_results = []
diff_results.append(adf_test(price_monthly_diff, "precio_promedio_global_diff"))
diff_results.append(kpss_test(price_monthly_diff, "precio_promedio_global_diff"))

diff_results_df = pd.DataFrame(diff_results)

print(diff_results_df)

# %%
# ============================================================
# 24. ACF Y PACF — SERIE GLOBAL ORIGINAL
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
plot_acf(price_monthly.dropna(), lags=24, ax=ax)
plt.title("ACF — Precio promedio global")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
plot_pacf(price_monthly.dropna(), lags=24, ax=ax, method="ywm")
plt.title("PACF — Precio promedio global")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 25. ACF Y PACF — SERIE GLOBAL DIFERENCIADA
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
plot_acf(price_monthly_diff.dropna(), lags=24, ax=ax)
plt.title("ACF — Primera diferencia del precio promedio")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
plot_pacf(price_monthly_diff.dropna(), lags=24, ax=ax, method="ywm")
plt.title("PACF — Primera diferencia del precio promedio")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 26. DESCOMPOSICIÓN DE LA SERIE TEMPORAL GLOBAL
# ============================================================

decomposition = seasonal_decompose(
    price_monthly,
    model="additive",
    period=12
)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.suptitle("Descomposición de la serie temporal del precio promedio", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 27. ESTACIONARIEDAD POR SEGMENTO
# ============================================================

segment_results = []

for segment in df_prices["segment_seed"].dropna().unique():
    segment_series = (
        df_prices[df_prices["segment_seed"] == segment]
        .groupby("month")["target_price"]
        .mean()
        .asfreq("MS")
        .interpolate(method="linear")
    )

    segment_results.append(adf_test(segment_series, f"precio_{segment}"))
    segment_results.append(kpss_test(segment_series, f"precio_{segment}"))

segment_stationarity_df = pd.DataFrame(segment_results)

print(segment_stationarity_df)

# %%
# ============================================================
# 28. SERIES TEMPORALES POR SEGMENTO
# ============================================================

plt.figure(figsize=(12, 6))

for segment, color in {
    "premium": "#0d3b66",
    "mid": "#ff4fa3",
    "mass": "#4ea8de"
}.items():

    segment_series = (
        df_prices[df_prices["segment_seed"] == segment]
        .groupby("month")["target_price"]
        .mean()
        .asfreq("MS")
        .interpolate(method="linear")
    )

    plt.plot(segment_series, label=segment, color=color, linewidth=2.5)

plt.title("Series temporales de precio promedio por segmento")
plt.xlabel("Fecha")
plt.ylabel("Precio promedio")
plt.legend(title="Segmento")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 29. ADF/KPSS POR PRODUCTO — ASIN CORREGIDO
# ============================================================

asin_results = []

for asin, group in df_prices.groupby("asin"):

    # Agrupamos por mes para evitar fechas duplicadas dentro del mismo ASIN
    series = (
        group
        .groupby("month")["target_price"]
        .mean()
        .sort_index()
    )

    # Forzamos frecuencia mensual solo después de quitar duplicados
    series = (
        series
        .asfreq("MS")
        .interpolate(method="linear")
        .dropna()
    )

    if len(series) >= 36:
        try:
            adf_res = adf_test(series, asin)
            kpss_res = kpss_test(series, asin)

            asin_results.append({
                "asin": asin,
                "adf_p_value": adf_res["p_value"],
                "adf_conclusion": adf_res["conclusion"],
                "kpss_p_value": kpss_res["p_value"],
                "kpss_conclusion": kpss_res["conclusion"],
                "n_obs": len(series)
            })

        except Exception:
            pass

asin_stationarity_df = pd.DataFrame(asin_results)

print("Productos analizados:", asin_stationarity_df.shape[0])

print("\nADF:")
print(asin_stationarity_df["adf_conclusion"].value_counts(normalize=True) * 100)

print("\nKPSS:")
print(asin_stationarity_df["kpss_conclusion"].value_counts(normalize=True) * 100)

print("\nPrimeras filas:")
print(asin_stationarity_df.head())

# %%
# ============================================================
# 30. RESUMEN VISUAL — ESTACIONARIEDAD POR PRODUCTO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.countplot(
    data=asin_stationarity_df,
    x="adf_conclusion",
    color="#0d3b66",
    ax=axes[0]
)
axes[0].set_title("Conclusión ADF por producto")
axes[0].set_xlabel("Resultado")
axes[0].set_ylabel("Número de productos")

sns.countplot(
    data=asin_stationarity_df,
    x="kpss_conclusion",
    color="#ff4fa3",
    ax=axes[1]
)
axes[1].set_title("Conclusión KPSS por producto")
axes[1].set_xlabel("Resultado")
axes[1].set_ylabel("Número de productos")

plt.tight_layout()
plt.show()

df["log_price"] = np.log(df["target_price"])
df["log_cpi_beauty"] = np.log(df["cpi_beauty"])

df["dlog_price"] = df["log_price"].diff()
# %%
# ============================================================
# 31. TRANSFORMACIÓN LOGARÍTMICA PARA EDA COMPLEMENTARIO
# ============================================================

df_prices["log_price"] = np.log(df_prices["target_price"])
df_prices["log_cpi"] = np.log(df_prices["cpi"])
df_prices["log_cpi_beauty"] = np.log(df_prices["cpi_beauty"])

df_prices = df_prices.replace([np.inf, -np.inf], np.nan)
df_prices = df_prices.dropna(subset=["log_price"])

print(df_prices["log_price"].describe())
# %%
# ============================================================
# 32. DISTRIBUCIÓN LOGARÍTMICA DE PRECIOS
# ============================================================

plt.figure(figsize=(11, 6))
sns.histplot(df_prices["log_price"], bins=50, kde=True, color="#ff4fa3")
plt.title("Distribución logarítmica de precios")
plt.xlabel("Log(Precio)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
# %%
# ============================================================
# 33. DISTRIBUCIÓN LOGARÍTMICA POR SEGMENTO
# ============================================================

plt.figure(figsize=(11, 6))
sns.histplot(
    data=df_prices,
    x="log_price",
    hue="segment_seed",
    bins=50,
    kde=True,
    element="step",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Distribución logarítmica por segmento")
plt.xlabel("Log(Precio)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
# %%
# ============================================================
# 34. BOXPLOT DE PRECIOS EN LOG POR SEGMENTO
# ============================================================

plt.figure(figsize=(11, 6))
sns.boxplot(
    data=df_prices,
    x="segment_seed",
    y="log_price",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)
plt.title("Boxplot de precios en log por segmento")
plt.xlabel("Segmento")
plt.ylabel("Log(Precio)")
plt.tight_layout()
plt.show()
# %%
# ============================================================
# 35. CORRELACIONES EN LOG Y DIFERENCIAS LOGARÍTMICAS
# ============================================================

macro_vars = ["cpi", "inflation_yoy", "interest_rate", "unemployment_rate"]

# Dataset mensual agregado
log_monthly = (
    df_prices
    .groupby("month")
    .agg({
        "log_price": "mean",
        "cpi": "mean",
        "inflation_yoy": "mean",
        "interest_rate": "mean",
        "unemployment_rate": "mean"
    })
    .reset_index()
)

# Correlación en log / niveles
corr_log = log_monthly[["log_price"] + macro_vars].corr()

# Diferencias
log_monthly["dlog_price"] = log_monthly["log_price"].diff()

for var in macro_vars:
    log_monthly[f"d_{var}"] = log_monthly[var].diff()

corr_diff = log_monthly[
    ["dlog_price"] + [f"d_{v}" for v in macro_vars]
].corr()

print("Correlación en log:\n", corr_log)
print("\nCorrelación en diferencias:\n", corr_diff)
# %%
# ============================================================
# 36. HEATMAP DE CORRELACIONES EN DIFERENCIAS
# ============================================================

plt.figure(figsize=(8, 6))

cmap = sns.diverging_palette(240, 330, as_cmap=True)

sns.heatmap(
    corr_diff,
    annot=True,
    cmap=cmap,
    center=0,
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlaciones en diferencias logarítmicas")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
# ============================================================
# 37. REGRESIÓN PRECIO VS TIPOS CON TENDENCIA
# ============================================================

import statsmodels.api as sm

# Dataset mensual
reg_data = macro_monthly.copy()

# Log del precio
reg_data["log_price"] = np.log(reg_data["target_price"])

# Tendencia temporal
reg_data["trend"] = np.arange(len(reg_data))

# Modelo 1: sin tendencia
X1 = sm.add_constant(reg_data["interest_rate"])
model1 = sm.OLS(reg_data["log_price"], X1).fit()

# Modelo 2: con tendencia
X2 = sm.add_constant(reg_data[["interest_rate", "trend"]])
model2 = sm.OLS(reg_data["log_price"], X2).fit()

print("Modelo sin tendencia:\n", model1.summary())
print("\nModelo con tendencia:\n", model2.summary())
# %%
# ============================================================
# 38. ESTACIONALIDAD MEDIA POR MES
# ============================================================

# Crear variable mes
df_prices["month_num"] = df_prices["month"].dt.month

# Promedio por mes (sobre todo el panel)
seasonality_month = (
    df_prices
    .groupby("month_num")["target_price"]
    .mean()
    .reset_index()
)

# Mapear nombres de meses
month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

seasonality_month["month_name"] = seasonality_month["month_num"].map(month_names)

# Orden correcto
seasonality_month = seasonality_month.sort_values("month_num")

# Gráfico
plt.figure(figsize=(11,6))

sns.lineplot(
    data=seasonality_month,
    x="month_name",
    y="target_price",
    marker="o",
    linewidth=2.5,
    color="#ff4fa3"
)

plt.title("Estacionalidad media de precios por mes")
plt.xlabel("Mes")
plt.ylabel("Precio promedio")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

seasonality_month_log = (
    df_prices
    .groupby("month_num")["log_price"]
    .mean()
    .reset_index()
)
# %%
# ============================================================
# 39. PROPORCIÓN DE VALORES IMPUTADOS POR SEGMENTO
# ============================================================

imputation_segment = (
    df
    .groupby("segment_seed")["was_imputed"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=imputation_segment,
    x="segment_seed",
    y="was_imputed",
    palette={
        "premium": "#0d3b66",
        "mid": "#ff4fa3",
        "mass": "#4ea8de"
    }
)

plt.title("Proporción de valores imputados por segmento")
plt.ylabel("Proporción imputada")
plt.xlabel("Segmento")

plt.tight_layout()
plt.show()

#Distribución de coberturas 
# %%
sns.histplot(df_prices["coverage_pct"], bins=30, color="#ff4fa3")
plt.title("Distribución de cobertura de datos por producto")
plt.xlabel("Cobertura (%)")
plt.show()

#Rachas de nulos
# %%
sns.histplot(df_prices["max_null_run"], bins=30, color="#0d3b66")
plt.title("Distribución de rachas máximas de valores nulos")
plt.xlabel("Meses consecutivos sin datos")
plt.show()