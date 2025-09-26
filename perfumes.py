import pandas as pd
import numpy as np
import re
import unicodedata

import pandas as pd

# rutas relativas (porque perfumes.py está en la misma carpeta)
PATH_FRA_PERFUMES = "fra_perfumes.csv"
PATH_FRA_CLEANED  = "fra_cleaned.csv"
PATH_EBAY_MEN     = "ebay_mens_perfume.csv"
PATH_EBAY_WOMEN   = "ebay_womens_perfume.csv"

# cargar
fra_perfumes = pd.read_csv(PATH_FRA_PERFUMES)
fra_cleaned  = pd.read_csv(PATH_FRA_CLEANED)
ebay_men     = pd.read_csv(PATH_EBAY_MEN)
ebay_women   = pd.read_csv(PATH_EBAY_WOMEN)


# Unimos eBay (hombres + mujeres)
ebay_raw = pd.concat([ebay_men, ebay_women], ignore_index=True)

# Si 'fra_cleaned' trae las columnas buenas, usamos esa; si no, caemos a fra_perfumes
def has_clean_cols(df):
    needed = {"Perfume","Brand","Rating Value","Rating Count","Year"}
    return needed.issubset(set(df.columns))

base = fra_cleaned.copy() if has_clean_cols(fra_cleaned) else fra_perfumes.copy()

# Para trabajar a gusto, nos quedamos con columnas relevantes si existen:
keep_cols = [
    "url", "Perfume", "Brand", "Country", "Gender",
    "Rating Value", "Rating Count", "Year",
    "Top", "Middle", "Base",
    "Perfumer1", "Perfumer2",
    "mainaccord1","mainaccord2","mainaccord3","mainaccord4","mainaccord5"
]
base = base[[c for c in keep_cols if c in base.columns]].copy()

# Renombramos a snake_case
rename_map = {
    "Perfume":"name", "Brand":"brand", "Country":"country", "Gender":"gender",
    "Rating Value":"rating_value", "Rating Count":"rating_count", "Year":"year",
    "Top":"notes_top", "Middle":"notes_middle", "Base":"notes_base",
    "Perfumer1":"perfumer1", "Perfumer2":"perfumer2",
    "mainaccord1":"accord1","mainaccord2":"accord2","mainaccord3":"accord3",
    "mainaccord4":"accord4","mainaccord5":"accord5"
}
base.rename(columns=rename_map, inplace=True)

# Tipados
for c in ["rating_value","rating_count","year"]:
    if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = strip_accents(s)
    # quitar tamaños y ediciones comunes
    s = re.sub(r"\b(\d+)\s?ml\b", " ", s)               # 50ml, 100 ml
    s = re.sub(r"\b(eau de parfum|eau de toilette|parfum|edt|edp|edc)\b", " ", s)
    s = re.sub(r"[\-_/.,:;!()&']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Sinónimos de marca comunes (añade más según veas en tu EDA)
brand_syn = {
    "christian dior": "dior",
    "ysl": "yves saint laurent",
    "yves-saint-laurent": "yves saint laurent",
    "jean paul gaultier": "jean paul gaultier",
    "giorgio armani": "armani",
    "armani": "armani",
}

def normalize_brand(s: str) -> str:
    t = normalize_text(s)
    return brand_syn.get(t, t)

# Fragrantica (base)
base["brand_norm"] = base["brand"].apply(normalize_brand)
base["name_norm"]  = base["name"].apply(normalize_text)




# Detectores de columnas
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

ebay = ebay_raw.copy()

col_brand = pick_col(ebay, ["brand","Brand","seller_brand","perfume_brand","Brand Name"])
col_name  = pick_col(ebay, ["name","title","product_name","Perfume","perfume_name","Item Title"])
col_price = pick_col(ebay, ["price","Price","current_price","selling_price","Converted Price","Item Price"])

if not all([col_brand, col_name, col_price]):
    raise ValueError(f"Necesito brand/name/price en eBay. Encontrado: brand={col_brand}, name={col_name}, price={col_price}")

ebay = ebay[[col_brand, col_name, col_price]].rename(columns={
    col_brand: "brand",
    col_name:  "name",
    col_price: "price_raw"
})

# limpiar precio: "€29,99", "EUR 45.00", "US $89.99", "69,00"
def to_price(x):
    if pd.isna(x): return np.nan
    s = str(x)
    # quitar moneda y texto
    s = re.sub(r"[^\d,.\-]", "", s)      # deja dígitos, coma y punto
    # si hay ambas coma y punto, asumimos formato europeo "1.234,56" -> "1234.56"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        # si solo hay coma, cambiar a punto
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

ebay["price"] = ebay["price_raw"].apply(to_price)

# Normalizados para matching
ebay["brand_norm"] = ebay["brand"].apply(normalize_brand)
ebay["name_norm"]  = ebay["name"].apply(normalize_text)

# Agregamos por (brand_norm, name_norm) con precio "robusto"
ebay_agg = (ebay
            .groupby(["brand_norm","name_norm"], as_index=False)
            .agg(price_median=("price","median"),
                 price_min=("price","min"),
                 price_max=("price","max"),
                 n_listings=("price","size")))

# (opcional) recorte de outliers por IQR en price_median
q1, q3 = ebay_agg["price_median"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
ebay_agg["price_median_clipped"] = ebay_agg["price_median"].clip(lower=low, upper=high)



merged_exact = base.merge(
    ebay_agg,
    on=["brand_norm","name_norm"],
    how="left"
)

hit_rate = merged_exact["price_median_clipped"].notna().mean()
print(f"Coverage exacto: {hit_rate:.1%}")



# Instala rapidfuzz si no está (Kaggle lo permite)
# !pip -q install rapidfuzz

from rapidfuzz import process, fuzz

unmatched = merged_exact[merged_exact["price_median_clipped"].isna()].copy()

# Preparamos diccionario: brand_norm -> lista de (name_norm, price, ...)
brand_to_ebay = {}
for b, grp in ebay_agg.groupby("brand_norm"):
    brand_to_ebay[b] = list(zip(grp["name_norm"], grp["price_median_clipped"]))

def best_match_within_brand(brand_norm, name_norm, min_score=90):
    cand = brand_to_ebay.get(brand_norm, [])
    if not cand:
        return (None, np.nan, 0)
    names = [c[0] for c in cand]
    # usamos ratio parcial que tolera prefijos/sufijos (EDP/EDT, etc.)
    match = process.extractOne(
        name_norm, names, scorer=fuzz.token_set_ratio
    )
    if match is None:
        return (None, np.nan, 0)
    best_name, score, idx = match[0], match[1], match[2]
    price = cand[idx][1]
    return (best_name, price, score)

res = unmatched.apply(
    lambda r: pd.Series(best_match_within_brand(r["brand_norm"], r["name_norm"], min_score=90),
                        index=["ebay_name_norm_match","price_fuzzy","fuzzy_score"]),
    axis=1
)

unmatched = pd.concat([unmatched.reset_index(drop=True), res], axis=1)

# Aceptamos matches con score >= 90 (ajusta a 85 si quieres más cobertura)
accepted = unmatched[unmatched["fuzzy_score"] >= 90].copy()

# Integramos los precios fuzzy en merged_exact donde falte
merged = merged_exact.copy()
mask = merged["price_median_clipped"].isna()
merged.loc[mask, "price_median_clipped"] = accepted.set_index(merged.loc[mask].index)["price_fuzzy"]


# Prioriza precio limpio y robusto
merged["price"] = merged["price_median_clipped"]

# Rangos fijos típicos (puedes ajustar a tu mercado)
def price_range(p):
    if pd.isna(p):
        return "unknown"
    if p < 30:   return "low"
    if p < 70:   return "medium"
    if p < 120:  return "high"
    return "luxury"

merged["price_range"] = merged["price"].apply(price_range)

# (Alternativa dinámica por cuantiles – útil si tu conjunto es “raro”)
def price_bucket_quantiles(s, qs=(0.25,0.5,0.75)):
    q1,q2,q3 = s.quantile(qs)
    def b(p):
        if pd.isna(p): return "unknown"
        if p <= q1: return "low"
        if p <= q2: return "medium"
        if p <= q3: return "high"
        return "luxury"
    return s.apply(b)

# merged["price_range"] = price_bucket_quantiles(merged["price"])

total = len(base)
with_price = merged["price"].notna().sum()
print(f"Perfumes con precio: {with_price}/{total} = {with_price/total:.1%}")

print(merged[["brand","name","price","price_range"]].head(10))

# guarda para posteriores notebooks/pasos
merged.to_csv("perfumes_merged_with_prices.csv", index=False)
