import os
import time
import socket
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import keepa

# =========================
# CONFIG
# =========================
KEEPA_KEY = os.getenv("KEEPA_KEY") or "jvhsdf2ij7fs558frni9394gvhd5flb9hgvrsmippet0fhj913fpe19hhk3ftg4a"

INPUT_CSV = "products_clean.csv"
OUTPUT_CSV = "keepa_monthly_history_raw.csv"
CHECKPOINT_CSV = "keepa_monthly_history_checkpoint.csv"
DEBUG_KEYS_CSV = "keepa_debug_data_keys.csv"

BATCH_SIZE = 25
SLEEP_BETWEEN_BATCHES = 2
SAVE_EVERY_BATCH = True

# Últimos 7 años, mensual
START_DATE = pd.Timestamp.today().normalize() - pd.DateOffset(years=7)
END_DATE = pd.Timestamp.today().normalize()

# =========================
# HELPERS
# =========================
def marketplace_to_domain(marketplace: str) -> str:
    m = str(marketplace).strip().upper()
    mapping = {
        "US": "US",
        "UK": "UK",
        "GB": "UK",
        "DE": "DE",
        "FR": "FR",
        "ES": "ES",
        "IT": "IT",
        "CA": "CA",
        "MX": "MX",
        "JP": "JP",
        "IN": "IN",
    }
    return mapping.get(m, "US")


def wait_for_connection(host: str = "api.keepa.com", sleep_seconds: int = 30):
    while True:
        try:
            socket.gethostbyname(host)
            return
        except Exception:
            print(f"Sin conexión con {host}. Reintentando en {sleep_seconds}s...")
            time.sleep(sleep_seconds)


def chunks(lst: List[dict], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def safe_get(d: Dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def first_existing_key(data: Dict, candidates: List[str]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for k in candidates:
        if k in data and f"{k}_time" in data:
            return k
    return None


def build_event_df(
    times,
    values,
    value_name: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp
) -> pd.DataFrame:
    if times is None or values is None:
        return pd.DataFrame(columns=["datetime", value_name])

    if len(times) == 0 or len(values) == 0:
        return pd.DataFrame(columns=["datetime", value_name])

    n = min(len(times), len(values))
    df = pd.DataFrame({
        "datetime": pd.to_datetime(times[:n], errors="coerce"),
        value_name: pd.to_numeric(values[:n], errors="coerce")
    }).dropna(subset=["datetime"])

    df = df[(df["datetime"] >= min_date) & (df["datetime"] <= max_date)].copy()
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return df


def monthly_last(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", value_col])

    out = (
        df.set_index("datetime")
          .resample("ME")[value_col]
          .last()
          .reset_index()
          .rename(columns={"datetime": "month"})
    )
    out["month"] = out["month"].dt.to_period("M").dt.to_timestamp()
    return out


def extract_product_monthly_panel(
    product: Dict,
    asin: str,
    brand_seed: str,
    category_seed: str,
    segment_seed: str,
    marketplace: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp
) -> pd.DataFrame:
    data = safe_get(product, "data", {})
    if not isinstance(data, dict) or len(data) == 0:
        return pd.DataFrame()

    amazon_key = first_existing_key(data, ["AMAZON", "NEW"])
    list_key = first_existing_key(data, ["LISTPRICE"])
    sales_key = first_existing_key(data, ["SALES", "SALESRANK"])

    amazon_df = pd.DataFrame(columns=["month", "amazon_price_raw"])
    list_df = pd.DataFrame(columns=["month", "list_price_raw"])
    sales_df = pd.DataFrame(columns=["month", "sales_rank_raw"])

    if amazon_key:
        ev = build_event_df(
            data.get(f"{amazon_key}_time"),
            data.get(amazon_key),
            "amazon_price_raw",
            min_date,
            max_date
        )
        amazon_df = monthly_last(ev, "amazon_price_raw")

    if list_key:
        ev = build_event_df(
            data.get(f"{list_key}_time"),
            data.get(list_key),
            "list_price_raw",
            min_date,
            max_date
        )
        list_df = monthly_last(ev, "list_price_raw")

    if sales_key:
        ev = build_event_df(
            data.get(f"{sales_key}_time"),
            data.get(sales_key),
            "sales_rank_raw",
            min_date,
            max_date
        )
        sales_df = monthly_last(ev, "sales_rank_raw")

    monthly_index = pd.date_range(
        start=min_date.to_period("M").to_timestamp(),
        end=max_date.to_period("M").to_timestamp(),
        freq="MS"
    )
    panel = pd.DataFrame({"month": monthly_index})

    for part in [amazon_df, list_df, sales_df]:
        if not part.empty:
            panel = panel.merge(part, on="month", how="left")

    panel["asin"] = asin
    panel["brand_seed"] = brand_seed
    panel["category_seed"] = category_seed
    panel["segment_seed"] = segment_seed
    panel["marketplace"] = marketplace

    panel["amazon_key_used"] = amazon_key
    panel["list_key_used"] = list_key
    panel["sales_key_used"] = sales_key

    if "amazon_price_raw" not in panel.columns:
        panel["amazon_price_raw"] = np.nan

    if "list_price_raw" not in panel.columns:
        panel["list_price_raw"] = np.nan

    if "sales_rank_raw" not in panel.columns:
        panel["sales_rank_raw"] = np.nan

    panel["amazon_price_final"] = panel["amazon_price_raw"] / 100.0
    panel["list_price_final"] = panel["list_price_raw"] / 100.0

    panel = panel.sort_values("month").reset_index(drop=True)
    return panel


def append_csv(df: pd.DataFrame, path: str):
    if df.empty:
        return
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)


def overwrite_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def get_done_asins_from_output(path: str) -> set:
    """
    Lee el CSV principal ya generado y recupera ASINs ya procesados.
    Esto permite reanudar sin duplicar aunque el script se haya pausado.
    """
    if not os.path.exists(path):
        return set()

    try:
        x = pd.read_csv(path, usecols=["asin"])
        return set(x["asin"].dropna().astype(str).str.strip().unique())
    except Exception as e:
        print(f"No se pudo leer {path} para reanudar: {e}")
        return set()


def save_checkpoint_asins(done_asins: set, path: str):
    """
    Guarda un checkpoint limpio: un ASIN por fila.
    """
    ckpt = pd.DataFrame({"asin": sorted(done_asins)})
    overwrite_csv(ckpt, path)


# =========================
# MAIN
# =========================
def main():
    if KEEPA_KEY == "TU_API_KEY_AQUI":
        raise ValueError("Configura tu KEEPA_KEY en entorno o dentro del script.")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"No existe {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = ["asin", "brand_seed", "category_seed", "segment_seed", "marketplace"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {INPUT_CSV}: {missing}")

    df = df[required].drop_duplicates(subset=["asin"]).copy()
    df["asin"] = df["asin"].astype(str).str.strip()
    df["marketplace"] = df["marketplace"].astype(str).str.strip().str.upper()

    # Reanudar desde lo ya guardado en la salida principal
    done_asins = get_done_asins_from_output(OUTPUT_CSV)

    # Si además existe un checkpoint previo, lo sumamos
    if os.path.exists(CHECKPOINT_CSV):
        try:
            ckpt = pd.read_csv(CHECKPOINT_CSV, usecols=["asin"])
            done_asins |= set(ckpt["asin"].dropna().astype(str).str.strip().unique())
        except Exception:
            pass

    total_original = len(df)
    df = df[~df["asin"].isin(done_asins)].copy().reset_index(drop=True)

    print(f"ASINs totales en input: {total_original}")
    print(f"ASINs ya procesados: {len(done_asins)}")
    print(f"ASINs pendientes por procesar: {len(df)}")

    if df.empty:
        print("No hay ASINs pendientes. Ya está todo procesado.")
        return

    # Para que te quede claro por dónde arranca
    starting_batch = (len(done_asins) // BATCH_SIZE) + 1
    print(f"Reanudando aproximadamente desde el lote {starting_batch} en adelante...")

    api = keepa.Keepa(KEEPA_KEY)
    debug_keys_rows = []

    global_batch_num = starting_batch - 1

    for marketplace, g in df.groupby("marketplace", sort=False):
        domain = marketplace_to_domain(marketplace)
        rows = g.to_dict("records")
        print(f"\nMarketplace {marketplace} | dominio {domain} | ASINs pendientes: {len(rows)}")

        for batch in chunks(rows, BATCH_SIZE):
            global_batch_num += 1
            wait_for_connection()

            asins = [r["asin"] for r in batch]
            meta = {r["asin"]: r for r in batch}

            print(f"  Lote {global_batch_num}: consultando {len(asins)} ASINs...")

            try:
                products = api.query(
                    asins,
                    domain=domain,
                    history=True,
                    rating=True,
                    stats=30,
                    progress_bar=False
                )
            except Exception as e:
                print(f"  -> Error en lote {global_batch_num}: {type(e).__name__}: {e}")
                traceback.print_exc()
                print("  -> Esperando 60s y seguimos...")
                time.sleep(60)
                continue

            batch_frames = []
            batch_done_asins = set()

            for product in products:
                asin = safe_get(product, "asin", None)
                if not asin:
                    continue

                asin = str(asin).strip()

                if asin not in meta:
                    continue

                m = meta[asin]

                data_obj = safe_get(product, "data", {})
                data_keys = list(data_obj.keys()) if isinstance(data_obj, dict) else []
                debug_keys_rows.append({
                    "asin": asin,
                    "brand_seed": m["brand_seed"],
                    "marketplace": m["marketplace"],
                    "data_keys": "|".join(sorted(data_keys))
                })

                try:
                    panel = extract_product_monthly_panel(
                        product=product,
                        asin=asin,
                        brand_seed=m["brand_seed"],
                        category_seed=m["category_seed"],
                        segment_seed=m["segment_seed"],
                        marketplace=m["marketplace"],
                        min_date=START_DATE,
                        max_date=END_DATE
                    )

                    # Marcar como procesado aunque venga vacío, para no atascarse siempre
                    batch_done_asins.add(asin)

                    if not panel.empty:
                        batch_frames.append(panel)

                except Exception as e:
                    print(f"    -> Error procesando ASIN {asin}: {type(e).__name__}: {e}")
                    batch_done_asins.add(asin)
                    continue

            # Guardar salida principal en append
            if batch_frames:
                out = pd.concat(batch_frames, ignore_index=True)

                if SAVE_EVERY_BATCH:
                    append_csv(out, OUTPUT_CSV)

                print(f"  -> Guardadas {len(out)} filas mensuales en {OUTPUT_CSV}")
            else:
                print("  -> Lote sin filas útiles")

            # Actualizar checkpoint limpio por ASIN
            done_asins |= batch_done_asins
            save_checkpoint_asins(done_asins, CHECKPOINT_CSV)

            time.sleep(SLEEP_BETWEEN_BATCHES)

    if debug_keys_rows:
        pd.DataFrame(debug_keys_rows).drop_duplicates().to_csv(DEBUG_KEYS_CSV, index=False)

    print("\nProceso terminado")
    print(f"Checkpoint limpio: {CHECKPOINT_CSV}")
    print(f"Salida principal: {OUTPUT_CSV}")
    print(f"Debug de claves: {DEBUG_KEYS_CSV}")


if __name__ == "__main__":
    main()