import pandas as pd
from pymongo import MongoClient

# =========================
# CONFIG
# =========================
CSV_PATH = "data/analytic_panel_final_FINAL.csv"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfm_cosmetics"
COLLECTION_NAME = "analytic_panel_final"

# =========================
# CARGA CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("CSV cargado correctamente")
print("Shape:", df.shape)
print("Columnas:", df.columns.tolist())

# Convertir fechas si existe la columna month
if "month" in df.columns:
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

# Reemplazar NaN por None para MongoDB
df = df.where(pd.notnull(df), None)

# Convertir a diccionarios
records = df.to_dict("records")

# =========================
# CONEXIÓN MONGO
# =========================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Limpia la colección anterior si existe
collection.delete_many({})

# Inserta datos
if records:
    collection.insert_many(records)

print("\nDatos cargados en MongoDB correctamente")
print("Base de datos:", DB_NAME)
print("Colección:", COLLECTION_NAME)
print("Documentos insertados:", collection.count_documents({}))

print("\nColecciones disponibles:")
print(db.list_collection_names())
