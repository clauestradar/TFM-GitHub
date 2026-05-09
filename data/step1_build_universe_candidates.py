import os
import re
import time
import math
import pandas as pd
from typing import Dict, List, Optional

import keepa

#CONFIG
KEEPA_KEY = os.getenv("KEEPA_KEY") or "PON_AQUI_TU_API_KEY"
INPUT_CSV = "keepa_seed_brands.csv"
OUTPUT_CSV = "products_universe_candidates.csv"

#Cuántos candidatos máximos intentar guardar por marca
MAX_PER_BRAND = 40

#Objetivo de universo candidato amplio para luego filtrar
TARGET_CANDIDATES_MIN = 1200

#HELPERS

BAD_TITLE_PATTERNS = [
    r"\bbook\b",
    r"\bbooks\b",
    r"\bpaperback\b",
    r"\bhardcover\b",
    r"\bnovel\b",
    r"\bmagazine\b",
    r"\bkindle\b",
    r"\bebook\b",
    r"\bjournal\b",
    r"\bnotebook\b",
    r"\bplanner\b",
    r"\btoothbrush\b",
    r"\btoothpaste\b",
    r"\brazor\b",
    r"\bshaver\b",
    r"\btrimmer\b",
    r"\bclipper\b",
    r"\bblade\b",
    r"\brefill blades\b",
    r"\breplacement\b",
    r"\bcharger\b",
    r"\bcable\b",
    r"\bcase\b",
    r"\baccessory\b",
    r"\bmachine\b",
    r"\bdevice\b",
    r"\btool\b",
    r"\bbrush set\b",
    r"\bempty bottle\b",
    r"\bpump bottle\b",
]

GOOD_COSMETIC_KEYWORDS = [
    "serum", "cleanser", "cream", "moisturizer", "moisturiser", "lotion",
    "gel", "balm", "toner", "essence", "ampoule", "mask", "sunscreen", "spf",
    "foundation", "concealer", "powder", "blush", "mascara", "eyeliner",
    "lipstick", "lip gloss", "lip balm", "makeup", "primer",
    "shampoo", "conditioner", "hair mask", "hair oil", "hair serum",
    "body wash", "body lotion", "body cream", "deodorant", "fragrance",
    "perfume", "eau de parfum", "eau de toilette", "cologne", "face wash",
    "retinol", "vitamin c", "niacinamide", "micellar", "exfoliant", "peel"
]

CATEGORY_HINTS = {
    "skincare": [
        "serum", "cleanser", "cream", "moisturizer", "moisturiser", "toner",
        "essence", "ampoule", "mask", "sunscreen", "spf", "retinol",
        "vitamin c", "niacinamide", "micellar", "face wash", "exfoliant"
    ],
    "makeup": [
        "foundation", "concealer", "powder", "blush", "mascara", "eyeliner",
        "lipstick", "lip gloss", "primer", "makeup"
    ],
    "haircare": [
        "shampoo", "conditioner", "hair mask", "hair oil", "hair serum",
        "leave in", "leave-in", "scalp", "hair"
    ],
    "hair_care": [
        "shampoo", "conditioner", "hair mask", "hair oil", "hair serum",
        "leave in", "leave-in", "scalp", "hair"
    ],
    "bodycare": [
        "body wash", "body lotion", "body cream", "deodorant", "soap", "body"
    ],
    "body_care": [
        "body wash", "body lotion", "body cream", "deodorant", "soap", "body"
    ],
    "fragrance": [
        "perfume", "fragrance", "eau de parfum", "eau de toilette", "cologne"
    ]
}

EXCLUDED_PRODUCT_GROUPS = {
    "Book", "Digital_Ebook_Purchase", "Office Product", "Electronics"
}


def normalize_text(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def looks_like_valid_asin(asin: str) -> bool:
    asin = str(asin).strip()
    return bool(re.fullmatch(r"[A-Z0-9]{10}", asin))


def contains_bad_title(title: str) -> bool:
    t = normalize_text(title)
    return any(re.search(p, t) for p in BAD_TITLE_PATTERNS)


def contains_good_cosmetic_signal(title: str) -> bool:
    t = normalize_text(title)
    return any(k in t for k in GOOD_COSMETIC_KEYWORDS)


def title_matches_seed_category(title: str, seed_category: str) -> bool:
    t = normalize_text(title)
    keywords = CATEGORY_HINTS.get(str(seed_category).lower(), [])
    if not keywords:
        return contains_good_cosmetic_signal(title)
    return any(k in t for k in keywords)


def safe_get(d: Dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def get_root_category_name(prod: Dict) -> str:
    cat = safe_get(prod, "rootCategory")
    if isinstance(cat, list) and len(cat) > 0:
        return str(cat[0])
    return str(cat) if cat is not None else ""


def marketplace_to_domain(marketplace: str) -> str:
    """
    Keepa domain helper. Ajusta si luego necesitas UK/DE/ES.
    """
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


def product_to_row(
    prod: Dict,
    brand_seed: str,
    category_seed: str,
    segment_seed: str,
    marketplace: str
) -> Dict:
    title = safe_get(prod, "title", "")
    brand_keepa = safe_get(prod, "brand", "")
    asin = safe_get(prod, "asin", "")
    product_group = safe_get(prod, "productGroup", "")
    root_category = get_root_category_name(prod)

    review_count = safe_get(prod, "reviewCount", None)
    rating = safe_get(prod, "rating", None)

    #Algunas cuentas lo devuelven como csv/list; aquí solo dejamos la señal básica
    images_csv = safe_get(prod, "imagesCSV", "")
    has_image = int(bool(images_csv))

    row = {
        "asin": asin,
        "brand_seed": brand_seed,
        "brand_keepa": brand_keepa,
        "title": title,
        "category_seed": category_seed,
        "segment_seed": segment_seed,
        "marketplace": marketplace,
        "product_group": product_group,
        "root_category": root_category,
        "review_count_snapshot": review_count,
        "rating_snapshot": rating,
        "has_image": has_image,
    }

    row["valid_asin"] = int(looks_like_valid_asin(asin))
    row["bad_title_flag"] = int(contains_bad_title(title))
    row["good_cosmetic_keyword_flag"] = int(contains_good_cosmetic_signal(title))
    row["category_match_flag"] = int(title_matches_seed_category(title, category_seed))
    row["excluded_product_group_flag"] = int(str(product_group) in EXCLUDED_PRODUCT_GROUPS)

    row["is_valid_candidate"] = int(
        row["valid_asin"] == 1
        and row["bad_title_flag"] == 0
        and row["excluded_product_group_flag"] == 0
        and (
            row["good_cosmetic_keyword_flag"] == 1
            or row["category_match_flag"] == 1
        )
    )

    score = 0
    score += 3 * row["category_match_flag"]
    score += 2 * row["good_cosmetic_keyword_flag"]
    score += 1 * row["has_image"]
    if pd.notna(review_count):
        score += 1
    if pd.notna(rating):
        score += 1
    row["priority_score"] = score

    if row["valid_asin"] == 0:
        row["exclusion_reason"] = "invalid_asin"
    elif row["bad_title_flag"] == 1:
        row["exclusion_reason"] = "bad_title_pattern"
    elif row["excluded_product_group_flag"] == 1:
        row["exclusion_reason"] = "excluded_product_group"
    elif row["is_valid_candidate"] == 0:
        row["exclusion_reason"] = "weak_cosmetic_signal"
    else:
        row["exclusion_reason"] = ""

    return row


#MAIN

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"No existe {INPUT_CSV}")

    if KEEPA_KEY == "PON_AQUI_TU_API_KEY":
        raise ValueError("Pon tu API key en KEEPA_KEY o expórtala en tu entorno con export KEEPA_KEY='...'.")

    seeds = pd.read_csv(INPUT_CSV)
    expected_cols = {"category", "segment", "brand", "marketplace"}
    missing = expected_cols - set(seeds.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {INPUT_CSV}: {missing}")

    api = keepa.Keepa(KEEPA_KEY)
    all_rows: List[Dict] = []

    print(f"Marcas semilla cargadas: {len(seeds)}")

    for i, seed in seeds.iterrows():
        brand = str(seed["brand"]).strip()
        category = str(seed["category"]).strip().lower()
        segment = str(seed["segment"]).strip().lower()
        marketplace = str(seed["marketplace"]).strip().upper()
        domain = marketplace_to_domain(marketplace)

        print(f"\n[{i+1}/{len(seeds)}] Buscando marca: {brand} | categoría: {category} | segmento: {segment}")

        try:
            finder_query = {
                "title": brand,
                "sort": [["current_SALES", "asc"]],
            }

            result = api.product_finder(
                finder_query,
                domain=domain,
                n_products=MAX_PER_BRAND
            )

            asins = list(result) if result is not None else []

            if not asins:
                print(f"  -> Sin resultados para {brand}")
                continue

            asins = list(dict.fromkeys(asins))[:MAX_PER_BRAND]
            print(f"  -> ASINs recuperados: {len(asins)}")

            products = api.query(
                asins,
                history=False,
                rating=True,
                stats=30,
                domain=domain,
                progress_bar=False
            )

            if not products:
                print("  -> Sin detalles de producto")
                continue

            for prod in products:
                row = product_to_row(
                    prod=prod,
                    brand_seed=brand,
                    category_seed=category,
                    segment_seed=segment,
                    marketplace=marketplace,
                )
                all_rows.append(row)

            time.sleep(1.2)

        except Exception as e:
            print(f"  -> Error con {brand}: {type(e).__name__}: {e}")
            continue

    if not all_rows:
        raise RuntimeError("No se recuperó ningún candidato. Revisa la API key, los tokens o la conectividad.")

    df = pd.DataFrame(all_rows)

    df = (
        df.sort_values(
            by=["is_valid_candidate", "priority_score", "review_count_snapshot", "rating_snapshot"],
            ascending=[False, False, False, False],
            na_position="last"
        )
        .drop_duplicates(subset=["asin"], keep="first")
        .reset_index(drop=True)
    )

    ordered_cols = [
        "asin",
        "brand_seed",
        "brand_keepa",
        "title",
        "category_seed",
        "segment_seed",
        "marketplace",
        "product_group",
        "root_category",
        "review_count_snapshot",
        "rating_snapshot",
        "has_image",
        "valid_asin",
        "bad_title_flag",
        "good_cosmetic_keyword_flag",
        "category_match_flag",
        "excluded_product_group_flag",
        "is_valid_candidate",
        "priority_score",
        "exclusion_reason",
    ]
    df = df[ordered_cols]

    df.to_csv(OUTPUT_CSV, index=False)

    n_total = len(df)
    n_valid = int(df["is_valid_candidate"].sum())

    print("\n================ RESUMEN ================")
    print(f"Candidatos únicos guardados: {n_total}")
    print(f"Candidatos válidos iniciales: {n_valid}")
    print(f"Archivo generado: {OUTPUT_CSV}")

    print("\nTop categorías seed:")
    print(df["category_seed"].value_counts(dropna=False).to_string())

    print("\nTop segmentos seed:")
    print(df["segment_seed"].value_counts(dropna=False).to_string())

    print("\nExclusiones más frecuentes:")
    print(df["exclusion_reason"].value_counts(dropna=False).head(10).to_string())

    if n_valid < TARGET_CANDIDATES_MIN:
        print(
            f"\nOjo: solo hay {n_valid} candidatos válidos iniciales. "
            f"Probablemente en el Paso 2 tendremos que ampliar búsqueda por marca."
        )
    else:
        print(f"\nBuen punto de partida: {n_valid} candidatos válidos iniciales.")


if __name__ == "__main__":
    main()