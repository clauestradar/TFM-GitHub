import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
START_DATE = "2018-01-01"
END_DATE = "2026-03-01"
OUTPUT_CSV = "macro_indicators_fred_v2.csv"

# =========================
# SERIES FRED
# =========================
SERIES = {
    "cpi": "CPIAUCSL",
    "cpi_beauty": "CUUR0000SEGB02",
    "unemployment_rate": "UNRATE",
    "interest_rate": "FEDFUNDS",
    "consumer_sentiment": "UMCSENT",
    "retail_sales": "RSXFS",
    "pce": "PCE",
    "sp500": "SP500",
    "housing_price_index": "CSUSHPISA",
    "credit_card_delinquency": "DRCCLACBS",
}

# =========================
# HELPERS
# =========================
def fred_csv_url(series_code: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"


def to_month_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_one_series(series_code: str, col_name: str) -> pd.DataFrame:
    url = fred_csv_url(series_code)
    print(f"Descargando {col_name} ({series_code})...")
    
    s = pd.read_csv(url)
    s.columns = ["date", col_name]
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s[col_name] = pd.to_numeric(s[col_name], errors="coerce")
    
    s = s.dropna(subset=["date"])
    s = s[(s["date"] >= pd.Timestamp(START_DATE)) & (s["date"] <= pd.Timestamp(END_DATE))]
    s = s.set_index("date")
    s = to_month_start(s)
    
    return s


def fetch_fred_series(series_dict):
    dfs = []
    for col_name, fred_code in series_dict.items():
        s = fetch_one_series(fred_code, col_name)
        dfs.append(s)
    out = pd.concat(dfs, axis=1).sort_index()
    return out


def add_yoy(df: pd.DataFrame, col: str, out_col: str):
    df[out_col] = df[col].pct_change(12) * 100
    return df


def classify_cycle(row):
    if pd.isna(row["unemployment_rate"]) or pd.isna(row["retail_sales_yoy"]) or pd.isna(row["consumer_sentiment"]):
        return np.nan

    if row["unemployment_rate"] >= 6 and row["retail_sales_yoy"] < 0:
        return "recession"
    elif row["unemployment_rate"] > 4.5 and row["consumer_sentiment"] < 75:
        return "slowdown"
    elif row["retail_sales_yoy"] > 2 and row["consumer_sentiment"] >= 85:
        return "expansion"
    else:
        return "recovery"


# =========================
# MAIN
# =========================
macro = fetch_fred_series(SERIES)

# rejilla mensual completa
monthly_index = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
macro = macro.reindex(monthly_index)

# imputación suave
macro = macro.interpolate(method="linear", limit_direction="both")
macro = macro.ffill().bfill()

# derivadas
macro = add_yoy(macro, "cpi", "inflation_yoy")
macro = add_yoy(macro, "retail_sales", "retail_sales_yoy")

macro["real_interest_rate"] = macro["interest_rate"] - macro["inflation_yoy"]
macro["post_pandemic_dummy"] = (macro.index >= pd.Timestamp("2021-01-01")).astype(int)
macro["high_rate_regime_dummy"] = (macro["interest_rate"] >= 4).astype(int)

macro["recession_dummy"] = (
    (macro["unemployment_rate"] >= 6) &
    (macro["retail_sales_yoy"] < 0)
).astype(int)

macro["cycle_phase"] = macro.apply(classify_cycle, axis=1)

macro = macro.reset_index().rename(columns={"index": "month"})

ordered_cols = [
    "month",
    "cpi",
    "cpi_beauty",
    "unemployment_rate",
    "interest_rate",
    "consumer_sentiment",
    "retail_sales",
    "pce",
    "sp500",
    "housing_price_index",
    "credit_card_delinquency",
    "inflation_yoy",
    "retail_sales_yoy",
    "real_interest_rate",
    "post_pandemic_dummy",
    "high_rate_regime_dummy",
    "recession_dummy",
    "cycle_phase",
]

macro = macro[ordered_cols]
macro.to_csv(OUTPUT_CSV, index=False)

print("\n Macro FRED reconstruido")
print(f"Archivo guardado: {OUTPUT_CSV}")
print(f"Shape: {macro.shape}")
print("\nPrimeras filas:")
print(macro.head())
print("\nNulos por columna:")
print(macro.isnull().mean())
print("\nRango temporal:")
print(macro['month'].min(), macro['month'].max())