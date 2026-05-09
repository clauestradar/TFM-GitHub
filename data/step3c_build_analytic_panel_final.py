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
#  FIX PRECIOS (INTELIGENTE)
# =========================
print("\nCorrigiendo escala de precios...")

df["amazon_price_raw"] = pd.to_numeric(df["amazon_price_raw"], errors="coerce")
df["list_price_raw"] = pd.to_numeric(df["list_price_raw"], errors="coerce")

# Detecta si están en centavos o ya escalados
df["amazon_price"] = np.where(
    df["amazon_price_raw"] > 1000,
    df["amazon_price_raw"] / 100,
    df["amazon_price_raw"]
)

df["list_price"] = np.where(
    df["list_price_raw"] > 1000,
    df["list_price_raw"] / 100,
    df["list_price_raw"]
)

# limpiar valores inválidos
df.loc[df["amazon_price"] <= 0, "amazon_price"] = np.nan
df.loc[df["list_price"] <= 0, "list_price"] = np.nan

print("Precio medio (check):", df["amazon_price"].mean())

# =========================
# 🧹 ELIMINAR DUPLICADOS (CRÍTICO)
# =========================
print("\nEliminando duplicados asin-month...")

df = (
    df.sort_values("month")
      .drop_duplicates(subset=["asin", "month"], keep="last")
)

# =========================
# MERGE MACRO
# =========================
print("\nMerge con macro...")

df = df.merge(macro, on="month", how="left")

# =========================
# FEATURE ENGINEERING (SAFE)
# =========================
print("\nCreando variables...")

# --- LOG PRECIO ---
df["price_log"] = np.where(
    df["amazon_price"] > 0,
    np.log(df["amazon_price"]),
    np.nan
)

# --- CAMBIO PRECIO ---
df = df.sort_values(["asin", "month"])

df["price_change_pct"] = (
    df.groupby("asin")["amazon_price"]
      .pct_change()
)

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
    df.groupby("asin")["sales_rank_raw"]
      .diff()
)

df["demand_proxy"] = 1 / df["sales_rank_raw"]

# --- MACRO ---
df["real_price"] = df["amazon_price"] / (df["cpi"] / 100)

# =========================
#  NO ELIMINAR FILAS
# =========================
print("\nPreservando panel completo...")

df = df.replace([np.inf, -np.inf], np.nan)

# =========================
#  VALIDACIÓN FINAL
# =========================
print("\nVerificando panel...")

check = df.groupby("asin")["month"].count()

print("ASINs:", df["asin"].nunique())
print("Min meses:", check.min())
print("Max meses:", check.max())

print("\nRango temporal:")
print(df["month"].min(), "→", df["month"].max())

print("\nPrecio stats:")
print(df["amazon_price"].describe())

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FINAL, index=False)

print("\n ANALYTIC PANEL FINAL LISTO")
print(f"Archivo: {OUTPUT_FINAL}")