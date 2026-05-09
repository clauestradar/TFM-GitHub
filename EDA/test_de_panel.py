import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chi2

# =========================
# 1. CARGAR DATOS
# =========================

df = pd.read_csv("analytic_panel_final_FINAL.csv")

df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.sort_values(["asin", "month"])

# =========================
# 2. CREAR VARIABLE LOG PRECIO
# =========================

if "price_log" not in df.columns:
    df["price_log"] = np.where(
        df["amazon_price"] > 0,
        np.log(df["amazon_price"]),
        np.nan
    )

# =========================
# 3. CREAR PANEL
# Filas = meses
# Columnas = productos ASIN
# Valores = log(precio)
# =========================

panel = df.pivot_table(
    index="month",
    columns="asin",
    values="price_log",
    aggfunc="mean"
)

# Orden temporal
panel = panel.sort_index()

print("Shape panel original:", panel.shape)

# =========================
# 4. FUNCIÓN FISHER-ADF PANEL
# =========================

def fisher_adf_panel_test(panel_data, min_obs=12, test_name="Fisher-ADF panel unit root test"):
    adf_pvalues = []

    for asin in panel_data.columns:
        series = panel_data[asin].dropna()

        if len(series) >= min_obs:
            try:
                result = adfuller(series, autolag="AIC")
                pvalue = result[1]

                if np.isfinite(pvalue) and pvalue > 0:
                    adf_pvalues.append(pvalue)

            except Exception:
                pass

    N = len(adf_pvalues)

    print("\n" + test_name)
    print("Número de series incluidas:", N)

    if N == 0:
        print("No hay suficientes series válidas para calcular el test.")
        return None

    fisher_stat = -2 * np.sum(np.log(adf_pvalues))
    p_value = 1 - chi2.cdf(fisher_stat, df=2 * N)

    print("Estadístico:", fisher_stat)
    print("p-valor:", p_value)

    if p_value < 0.05:
        conclusion = "Se rechaza H0 de raíz unitaria. El panel muestra evidencia de estacionariedad."
    else:
        conclusion = "No se rechaza H0 de raíz unitaria. El panel muestra evidencia de no estacionariedad."

    print("Conclusión:", conclusion)

    return {
        "test": test_name,
        "n_series": N,
        "statistic": fisher_stat,
        "p_value": p_value,
        "conclusion": conclusion
    }

# =========================
# 5. TEST EN NIVELES
# =========================

result_levels = fisher_adf_panel_test(
    panel_data=panel,
    min_obs=36,
    test_name="Fisher-ADF panel unit root test — niveles"
)

# =========================
# 6. TEST EN PRIMERA DIFERENCIA
# =========================

panel_diff = panel.diff()

print("\nShape panel diferenciado:", panel_diff.shape)

result_diff = fisher_adf_panel_test(
    panel_data=panel_diff,
    min_obs=12,
    test_name="Fisher-ADF panel unit root test — primera diferencia"
)

# =========================
# 7. TABLA RESUMEN
# =========================

results = [r for r in [result_levels, result_diff] if r is not None]

if results:
    results_df = pd.DataFrame(results)
    print("\nTabla resumen:")
    print(results_df)

    results_df.to_csv("panel_unit_root_tests_results.csv", index=False)
    print("\nArchivo guardado: panel_unit_root_tests_results.csv")