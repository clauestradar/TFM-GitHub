import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("keepa_monthly_history_raw.csv")

print("Filas totales:", len(df))
print("ASINs únicos:", df["asin"].nunique())

# =========================
# 1. FILTRO: 7 AÑOS COMPLETOS
# =========================
obs = df.groupby("asin")["month"].nunique()

asins_85 = obs[obs == 85].index
df = df[df["asin"].isin(asins_85)].copy()

print("ASINs con 85 meses:", df["asin"].nunique())

# =========================
# 2. CALIDAD DE PRECIO
# =========================

quality = df.groupby("asin").agg(
    n_months=("month", "nunique"),
    n_price_obs=("amazon_price_final", lambda x: x.notna().sum()),
)

quality["price_coverage"] = quality["n_price_obs"] / 85

print("\nResumen calidad:")
print(quality.describe())

# =========================
# 3. FILTRO DE CALIDAD
# =========================

MIN_PRICE_OBS = 36  # mínimo defendible

good_asins = quality[
    quality["n_price_obs"] >= MIN_PRICE_OBS
].index

df = df[df["asin"].isin(good_asins)].copy()

print("ASINs tras filtro calidad:", df["asin"].nunique())

# =========================
# 4. META INFO
# =========================

meta = df[[
    "asin",
    "brand_seed",
    "category_seed",
    "segment_seed"
]].drop_duplicates()

# =========================
# 5. RANKING DE CALIDAD
# =========================

ranking = quality.loc[good_asins].sort_values(
    ["n_price_obs", "price_coverage"],
    ascending=False
)

# =========================
# 6. SELECCIÓN BALANCEADA
# =========================

TARGET_TOTAL = 500
TARGET_PER_SEGMENT = TARGET_TOTAL // 3

final_asins = []

for segment in ["mass", "mid", "premium"]:
    asins_segment = meta[meta["segment_seed"] == segment]["asin"]

    ranked_segment = ranking.loc[asins_segment].sort_values(
        ["n_price_obs", "price_coverage"],
        ascending=False
    )

    selected = ranked_segment.head(TARGET_PER_SEGMENT).index.tolist()
    final_asins.extend(selected)

# Si faltan algunos para llegar a 500
remaining = TARGET_TOTAL - len(final_asins)

if remaining > 0:
    extra = ranking.drop(index=final_asins).head(remaining).index.tolist()
    final_asins.extend(extra)

df_final = df[df["asin"].isin(final_asins)].copy()

print("\nASINs finales:", df_final["asin"].nunique())

# =========================
# 7. CHECK FINAL
# =========================

check = df_final.groupby("asin").agg(
    n_months=("month", "nunique"),
    n_price_obs=("amazon_price_final", lambda x: x.notna().sum()),
)

print("\nVerificación final:")
print(check.describe())

# =========================
# 8. GUARDAR
# =========================

df_final.to_csv("final_panel_500.csv", index=False)

print("\nDataset final guardado como: final_panel_500.csv")