import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_PANEL = "final_panel_500.csv"
INPUT_MACRO = "macro_indicators_fred_v2.csv"
OUTPUT_FINAL = "analytic_panel_final.csv"

# =========================
# LOAD
# =========================
print("Cargando datos...")

df = pd.read_csv(INPUT_PANEL)
macro = pd.read_csv(INPUT_MACRO)

df["month"] = pd.to_datetime(df["month"])
macro["month"] = pd.to_datetime(macro["month"])

print("Shape inicial:", df.shape)

# =========================
# FIX PRECIOS (CRÍTICO)
# =========================
print("\nCorrigiendo escala de precios...")

# convertir a numérico
df["amazon_price_raw"] = pd.to_numeric(df["amazon_price_raw"], errors="coerce")
df["list_price_raw"] = pd.to_numeric(df["list_price_raw"], errors="coerce")

# 🔥 REESCALADO CORRECTO
df["amazon_price"] = df["amazon_price_raw"] / 100
df["list_price"] = df["list_price_raw"] / 100

# sanity check: eliminar valores absurdos
df.loc[df["amazon_price"] <= 0, "amazon_price"] = np.nan
df.loc[df["list_price"] <= 0, "list_price"] = np.nan

print("Precio medio:", df["amazon_price"].mean())

# =========================
# MERGE MACRO
# =========================
print("\nMerge con macro...")

df = df.merge(macro, on="month", how="left")

# =========================
# FEATURE ENGINEERING (SIN ROMPER PANEL)
# =========================
print("\nCreando variables...")

# --- LOG PRECIO (solo donde exista) ---
df["price_log"] = np.where(
    df["amazon_price"] > 0,
    np.log(df["amazon_price"]),
    np.nan
)

# --- CAMBIO PRECIO ---
df["price_change_pct"] = (
    df.sort_values("month")
      .groupby("asin")["amazon_price"]
      .pct_change()
)

# winsorización suave
df["price_change_pct"] = df["price_change_pct"].clip(-1, 1)

# --- DESCUENTO ---
df["discount_rate"] = (
    (df["list_price"] - df["amazon_price"]) /
    df["list_price"]
)

# --- POSICIONAMIENTO ---
df["category_mean_price"] = (
    df.groupby(["month", "category_seed"])["amazon_price"]
      .transform("mean")
)

df["price_vs_category_mean"] = (
    df["amazon_price"] / df["category_mean_price"]
)

# --- SALES RANK ---
df["sales_rank_raw"] = pd.to_numeric(df["sales_rank_raw"], errors="coerce")

df.loc[df["sales_rank_raw"] <= 0, "sales_rank_raw"] = np.nan

df["sales_rank_log"] = np.log(df["sales_rank_raw"])

df["sales_rank_change"] = (
    df.sort_values("month")
      .groupby("asin")["sales_rank_raw"]
      .diff()
)

df["demand_proxy"] = 1 / df["sales_rank_raw"]

# --- MACRO ---
df["real_price"] = df["amazon_price"] / (df["cpi"] / 100)

# =========================
#  NO ELIMINAR FILAS
# =========================
print("\nPreservando panel completo...")

# solo limpiamos infinitos
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# VERIFICACIÓN PANEL
# =========================
print("\nVerificando panel...")

check = df.groupby("asin")["month"].count()

print("ASINs:", df["asin"].nunique())
print("Min meses:", check.min())
print("Max meses:", check.max())

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FINAL, index=False)

print("\n ANALYTIC PANEL FINAL LISTO")
print(f"Archivo: {OUTPUT_FINAL}")

# =========================
# RESUMEN
# =========================
print("\nResumen:")

print("Shape:", df.shape)
print("ASINs:", df["asin"].nunique())
print("Meses:", df["month"].min(), "→", df["month"].max())

print("\nPrecio stats:")
print(df["amazon_price"].describe())