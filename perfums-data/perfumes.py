# ============================================================
# PERFUMES — Integración Fragrantica + eBay (precios)
# Objetivo:
#   1) Cargar robustamente 4 CSV (distintos encodings/separadores)
#   2) Normalizar marca y nombre para matching
#   3) Unir eBay (men+women) y agregar precios robustos por producto
#   4) Matching exacto + fuzzy dentro de marca (índices alineados)
#   5) Crear columnas price y price_range y guardar dataset final
# ============================================================

# -------------------------
# 0) IMPORTS
# -------------------------
import pandas as pd
import numpy as np
import re
import unicodedata

# Opcional: fuzzy matching (si no está instalado, saltamos la parte fuzzy)
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False
    print("[AVISO] 'rapidfuzz' no está instalado. Se hará solo matching exacto. "
          "Instala con: pip install rapidfuzz")

# -------------------------
# 1) RUTAS (ajusta si mueves el script)
# -------------------------
PATH_FRA_PERFUMES = "fra_perfumes.csv"
PATH_FRA_CLEANED  = "fra_cleaned.csv"
PATH_EBAY_MEN     = "ebay_mens_perfume.csv"
PATH_EBAY_WOMEN   = "ebay_womens_perfume.csv"

# -------------------------
# 2) LECTURA ROBUSTA
#    Maneja UTF-8/latin1/cp1252 y autodetección de separador
# -------------------------
def read_csv_robust(path, nrows=None):
    attempts = [
        dict(),  # utf-8
        dict(encoding="utf-8-sig"),
        dict(encoding="latin1"),
        dict(encoding="cp1252"),
        dict(sep=None, engine="python"),
        dict(sep=None, engine="python", encoding="latin1"),
        dict(sep=None, engine="python", encoding="cp1252"),
        # pandas >=2.0
        dict(encoding="latin1", encoding_errors="ignore"),
        dict(sep=None, engine="python", encoding="latin1", encoding_errors="ignore"),
    ]
    last_err = None
    for kw in attempts:
        try:
            return pd.read_csv(path, nrows=nrows, **kw)
        except Exception as e:
            last_err = e
    raise last_err

fra_perfumes = read_csv_robust(PATH_FRA_PERFUMES)
fra_cleaned  = read_csv_robust(PATH_FRA_CLEANED)
ebay_men     = read_csv_robust(PATH_EBAY_MEN)
ebay_women   = read_csv_robust(PATH_EBAY_WOMEN)

# -------------------------
# 3) BASE PRINCIPAL: preferimos fra_cleaned si ya trae columnas limpias
# -------------------------
def has_clean_cols(df):
    needed = {"Perfume","Brand","Rating Value","Rating Count","Year"}
    return needed.issubset(set(df.columns))

base = fra_cleaned.copy() if has_clean_cols(fra_cleaned) else fra_perfumes.copy()

keep_cols = [
    "url", "Perfume", "Brand", "Country", "Gender",
    "Rating Value", "Rating Count", "Year",
    "Top", "Middle", "Base",
    "Perfumer1", "Perfumer2",
    "mainaccord1","mainaccord2","mainaccord3","mainaccord4","mainaccord5"
]
base = base[[c for c in keep_cols if c in base.columns]].copy()

rename_map = {
    "Perfume":"name", "Brand":"brand", "Country":"country", "Gender":"gender",
    "Rating Value":"rating_value", "Rating Count":"rating_count", "Year":"year",
    "Top":"notes_top", "Middle":"notes_middle", "Base":"notes_base",
    "Perfumer1":"perfumer1", "Perfumer2":"perfumer2",
    "mainaccord1":"accord1","mainaccord2":"accord2","mainaccord3":"accord3",
    "mainaccord4":"accord4","mainaccord5":"accord5"
}
base.rename(columns=rename_map, inplace=True)

for c in ["rating_value","rating_count","year"]:
    if c in base.columns:
        base[c] = pd.to_numeric(base[c], errors="coerce")

# -------------------------
# 4) UNIR EBAY (men + women) y PREPARAR CAMPOS
# -------------------------
ebay_raw = pd.concat([ebay_men, ebay_women], ignore_index=True)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_brand = pick_col(ebay_raw, ["brand","Brand","seller_brand","perfume_brand","Brand Name"])
col_name  = pick_col(ebay_raw, ["name","title","product_name","Perfume","perfume_name","Item Title"])
col_price = pick_col(ebay_raw, ["price","Price","current_price","selling_price","Converted Price","Item Price"])

if not all([col_brand, col_name, col_price]):
    raise ValueError(f"[ERROR] No encuentro columnas brand/name/price en eBay. "
                     f"brand={col_brand}, name={col_name}, price={col_price}")

ebay = ebay_raw[[col_brand, col_name, col_price]].rename(columns={
    col_brand: "brand",
    col_name:  "name",
    col_price: "price_raw"
})

# Limpieza robusta de precio desde string a float
def to_price(x):
    if pd.isna(x): 
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d,.\-]", "", s)  # deja dígitos/coma/punto
    # Caso mixto coma+punto: formato europeo "1.234,56" -> "1234.56"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

ebay["price"] = ebay["price_raw"].apply(to_price)

# -------------------------
# 5) NORMALIZACIÓN PARA MATCHING (marca y nombre)
# -------------------------
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

# Tokens comunes a eliminar de los nombres (mejora cobertura exacta)
BAD_TOKENS = r"\b(eau de parfum|eau de toilette|parfum|edt|edp|edc|tester|gift set|set|limited edition|intense|extreme|pride|collector|spray|refill|mini|travel|vaporisateur)\b"

def normalize_text(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s).lower().strip()
    s = strip_accents(s)
    s = re.sub(r"\b(\d+)\s?ml\b", " ", s)         # 50ml, 100 ml
    s = re.sub(BAD_TOKENS, " ", s)                # tokens no distintivos
    s = re.sub(r"[\-_/.,:;!()&+']", " ", s)       # separadores y signos
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Sinónimos de marca (extiende según tu EDA)
brand_syn = {
    "christian dior": "dior",
    "dior": "dior",
    "ysl": "yves saint laurent",
    "yves-saint-laurent": "yves saint laurent",
    "jean paul gaultier": "jean paul gaultier",
    "guerlain": "guerlain",
    "armani": "armani",
    "giorgio armani": "armani",
    "dolce gabbana": "dolce gabbana",
    "d&g": "dolce gabbana",
    "victor and rolf": "viktor rolf",
    "viktor&rolf": "viktor rolf",
}

def normalize_brand(s: str) -> str:
    t = normalize_text(s)
    return brand_syn.get(t, t)

# Aplica normalización en ambos mundos
base["brand_norm"] = base["brand"].apply(normalize_brand)
base["name_norm"]  = base["name"].apply(normalize_text)

ebay["brand_norm"] = ebay["brand"].apply(normalize_brand)
ebay["name_norm"]  = ebay["name"].apply(normalize_text)

# -------------------------
# 6) AGREGAR PRECIOS EN EBAY POR (brand_norm, name_norm)
#    Usamos mediana y recortamos outliers por IQR
# -------------------------
ebay_agg = (ebay
            .groupby(["brand_norm","name_norm"], as_index=False)
            .agg(price_median=("price","median"),
                 price_min=("price","min"),
                 price_max=("price","max"),
                 n_listings=("price","size")))

if not ebay_agg.empty and ebay_agg["price_median"].notna().any():
    q1, q3 = ebay_agg["price_median"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    ebay_agg["price_median_clipped"] = ebay_agg["price_median"].clip(lower=low, upper=high)
else:
    ebay_agg["price_median_clipped"] = np.nan

# ---------- 6bis) Funciones extra de normalización ----------
YEAR_TOKENS = r"\b(19\d{2}|20\d{2})\b"   # 1980..2099, limpia años del nombre
GENDER_TOKENS = r"\b(for (men|him)|for (women|her)|men|man|women|woman|male|female|unisex)\b"

def normalize_name_stronger(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(YEAR_TOKENS, " ", s)
    s = re.sub(r"\(.*?\)", " ", s)           # quita paréntesis y su contenido
    s = re.sub(GENDER_TOKENS, " ", s)
    s = re.sub(r"\b(limited|exclusive|collector|tester|intense|extreme|elixir|noir|sport|fresh|summer|night|day|pride|holiday|gift|set)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def core_tokens(name: str, top_k: int = 3) -> str:
    """Elige tokens 'fuertes': sin palabras muy cortas, quita stop-words olfativas comunes.
       Devuelve los primeros K tokens “representativos” concatenados (orden estable)."""
    if not name:
        return ""
    toks = name.split()
    # stop-words sencillas (añade más según veas)
    stop = set(["eau","de","le","la","the","and","pour","for","with","by",
                "intense","extreme","elixir","noir","sport","fresh","summer",
                "edt","edp","edc","parfum","toilette","spray","refill",
                "tester","set","gift","mini","travel"])
    toks = [t for t in toks if len(t) > 2 and t not in stop]
    return " ".join(toks[:top_k])

# Aplica normalización fuerte y core_name
base["name_strong"] = base["name"].apply(normalize_name_stronger)
ebay["name_strong"] = ebay["name"].apply(normalize_name_stronger)

base["core_name"] = base["name_strong"].apply(lambda s: core_tokens(s, top_k=3))
ebay["core_name"] = ebay["name_strong"].apply(lambda s: core_tokens(s, top_k=3))

# Reagrega eBay por (brand_norm, name_norm) ya lo teníamos, añadimos por core_name:
ebay_agg_full = ebay_agg.merge(
    ebay[["brand_norm","name_norm","name_strong","core_name"]].drop_duplicates(),
    on=["brand_norm","name_norm"],
    how="left"
)

# ---------- 7bis) MATCHING EXACTO ampliado ----------
# 7.1 exacto clásico (brand_norm + name_norm)
merged_exact = base.merge(
    ebay_agg_full,
    on=["brand_norm","name_norm"],
    how="left",
    suffixes=("","_ebay")
)

# 7.2 exacto por (brand_norm + core_name) para los no matcheados
mask_missing = merged_exact["price_median_clipped"].isna()
if mask_missing.any():
    aux = base.loc[mask_missing, ["brand_norm","core_name"]].merge(
        ebay_agg_full[["brand_norm","core_name","price_median_clipped"]],
        on=["brand_norm","core_name"],
        how="left"
    )
    merged_exact.loc[mask_missing, "price_median_clipped"] = aux["price_median_clipped"].values

print(f"Coverage exacto (ampliado): {merged_exact['price_median_clipped'].notna().mean():.1%}")

# ---------- 8bis) MATCHING DIFUSO mejorado ----------
merged = merged_exact.copy()
if HAS_RAPIDFUZZ:
    from rapidfuzz import process, fuzz

    mask_unmatched = merged["price_median_clipped"].isna()
    unmatched = merged.loc[mask_unmatched, ["brand_norm","name_strong","core_name"]].copy()

    # Índices por marca
    brand_to_ebay = {}
    for b, grp in ebay_agg_full.groupby("brand_norm"):
        brand_to_ebay[b] = grp[["name_strong","core_name","price_median_clipped"]].dropna(subset=["price_median_clipped"]).values.tolist()
        # => lista de [name_strong, core_name, price]

    def best_fuzzy(brand_norm, query_name, query_core, min_score=82):
        cands = brand_to_ebay.get(brand_norm, [])
        if not cands:
            return pd.Series({"price_fuzzy": np.nan, "fuzzy_score": 0, "how":"none"})

        # 1) fuzzy sobre name_strong (token_set_ratio)
        names = [c[0] for c in cands]
        match = process.extractOne(query_name, names, scorer=fuzz.token_set_ratio)
        best_price, best_score, how = np.nan, 0, "none"
        if match is not None:
            idx = match[2]
            best_price = cands[idx][2]
            best_score = match[1]
            how = "fuzzy_name"

        # 2) contención de tokens (core ⊆ name_strong) como plan B
        if (best_score < min_score) and query_core:
            qset = set(query_core.split())
            for n_str, c_core, pr in cands:
                nset = set(str(n_str).split())
                # si todos los tokens del core del query están en el candidato, acepta
                if qset.issubset(nset):
                    best_price, best_score, how = pr, 100, "token_subset"
                    break
        if best_score >= min_score or how == "token_subset":
            return pd.Series({"price_fuzzy": best_price, "fuzzy_score": best_score, "how": how})
        return pd.Series({"price_fuzzy": np.nan, "fuzzy_score": best_score, "how": how})

    # ¡No resetees índice!
    fuzzy_res = unmatched.apply(
        lambda r: best_fuzzy(r["brand_norm"], r["name_strong"], r["core_name"], min_score=82), axis=1
    )
    to_assign = fuzzy_res["price_fuzzy"].notna()
    assign_idx = unmatched.index[to_assign]

    merged.loc[assign_idx, "price_median_clipped"] = fuzzy_res.loc[assign_idx, "price_fuzzy"].values

    # Diagnóstico
    print("Fuzzy asignados:", int(to_assign.sum()))
    print(f"Cobertura total tras fuzzy: {merged['price_median_clipped'].notna().mean():.1%}")
else:
    print("[INFO] Saltando matching difuso (rapidfuzz no disponible).")

# ---------- 9) PRECIO FINAL + RANGOS ----------
merged["price"] = pd.to_numeric(merged["price_median_clipped"], errors="coerce")

def price_range(p):
    if pd.isna(p): return "unknown"
    if p < 30:   return "low"
    if p < 70:   return "medium"
    if p < 120:  return "high"
    return "luxury"

merged["price_range"] = merged["price"].apply(price_range)

# ---------- 10) Checks y export ----------
total = len(base)
with_price = merged["price"].notna().sum()
print(f"Perfumes con precio: {with_price}/{total} = {with_price/total:.1%}")

print("\nEjemplos (brand, name, price, price_range):")
print(merged[["brand","name","price","price_range"]].head(10))

merged.to_csv("perfumes_merged_with_prices.csv", index=False, encoding="utf-8")
print("\n[OK] Guardado: perfumes_merged_with_prices.csv")
