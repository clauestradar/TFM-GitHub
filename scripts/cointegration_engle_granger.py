import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

# =========================
# 1. Cargar dataset
# =========================

df = pd.read_csv("/Users/clau/Desktop/Máster Data Analytics /TFM/Código TFM/analytic_panel_final_FINAL.csv")

# Asegurar formato fecha
df["month"] = pd.to_datetime(df["month"])

# =========================
# 2. Crear serie mensual agregada
# =========================
# Precio promedio mensual del panel

monthly = (
    df.groupby("month")
    .agg({
        "amazon_price": "mean",
        "inflation_yoy": "mean",
        "unemployment_rate": "mean",
        "interest_rate": "mean",
        "cpi": "mean",
        "cpi_beauty": "mean",
        "real_interest_rate": "mean"
    })
    .reset_index()
    .sort_values("month")
)

# =========================
# 3. Transformación logarítmica del precio
# =========================

monthly["log_price"] = np.log(monthly["amazon_price"])

# Eliminar posibles infinitos o nulos
monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# 4. Test Engle-Granger
# =========================

variables_macro = [
    "inflation_yoy",
    "unemployment_rate",
    "interest_rate",
    "cpi",
    "cpi_beauty",
    "real_interest_rate"
]

results = []

for var in variables_macro:
    score, p_value, critical_values = coint(monthly["log_price"], monthly[var])

    if p_value < 0.05:
        conclusion = "Rechazo H0: existe cointegración"
    else:
        conclusion = "No rechazo H0: no hay evidencia de cointegración"

    results.append({
        "Variable macroeconómica": var,
        "Estadístico Engle-Granger": score,
        "p-value": p_value,
        "Valor crítico 1%": critical_values[0],
        "Valor crítico 5%": critical_values[1],
        "Valor crítico 10%": critical_values[2],
        "Conclusión": conclusion
    })

results_df = pd.DataFrame(results)

# =========================
# 5. Guardar resultados
# =========================

results_df.to_csv("outputs/tables/engle_granger_cointegration_results.csv", index=False)

print("\nRESULTADOS TEST DE COINTEGRACIÓN ENGLE-GRANGER\n")
print(results_df)