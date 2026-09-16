"""
cleaning.py
===========
Normalización y clasificación de la base consolidada de pozos
(Sistema Acuífero Raigón + Sistema Acuífero Guaraní).

Puntos de criterio de dominio (documentados para revisar con Georgina):
- La base NO trae una columna "Acuífero" explícita. Se clasifica por
  `Departamento`: San José -> Raigón; Tacuarembó/Rivera/Salto -> Guaraní.
  12 pozos sin Departamento quedan como "Sin clasificar".
- Como dato complementario (no como clasificador principal, porque un
  mismo pozo puede tener varias formaciones a distintas profundidades),
  se puede mirar la Formación geológica relevada en Litología.
- "Tiene dato" en Hidráulica/Litología/Hidroquímica/Histórico se define
  como "aparece al menos una vez para ese Cod UK en esa hoja", sin juzgar
  calidad/completitud del dato en sí.
"""
from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Clasificación por acuífero (vía Departamento)
# ---------------------------------------------------------------------------
DEPARTAMENTO_A_ACUIFERO = {
    "san josé": "Raigón",
    "tacuarembó": "Guaraní",
    "rivera": "Guaraní",
    "salto": "Guaraní",
}

# Formaciones geológicas asociadas a cada acuífero (uso complementario,
# por ejemplo para el filtro de "formación perforada" dentro de Litología).
FORMACIONES_RAIGON = {"Fm Raigon", "Fm Dolores", "Fm Chuy", "Fm Libertad", "Fm Camacho"}
FORMACIONES_GUARANI = {
    "Fm Tacuarembo", "Fm Arapey", "Fm Rivera", "Fm Buena Vista", "Fm Yaguari",
    "Fm Las Arenas", "Fm Fray Bentos", "Fm Paso Aguiar", "Fm Tres Islas",
    "Fm Cerro pelado", "Fm San Gregorio", "Fm Cerrezuelo", "Fm  Fraile Muerto",
}

_SGRH_RE = re.compile(r"SGRH-[\w.-]+", flags=re.IGNORECASE)


# Frontera de latitud entre el cluster sur (Raigón, San José) y el cluster
# norte (Guaraní, Tacuarembó/Rivera/Salto). Hay un vacío real de pozos entre
# aprox. -34.3 y -32.0, así que -33.0 separa ambos grupos sin ambigüedad.
LATITUD_LIMITE_RAIGON_GUARANI = -33.0


def normalizar_pozos(pozos: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia texto/casing de columnas clave y agrega `acuifero_departamento`
    (clasificación cruda por texto de Departamento, con errores de carga
    conocidos — ver README). La columna `acuifero` "oficial" se calcula
    después, por ubicación, en `clasificar_acuifero_por_ubicacion`.
    """
    df = pozos.copy()

    df["Departamento"] = df["Departamento"].astype("string").str.strip()
    dep_norm = df["Departamento"].str.lower()
    df["acuifero_departamento"] = dep_norm.map(DEPARTAMENTO_A_ACUIFERO).fillna("Sin clasificar")

    df["Fuente BD"] = df["Fuente BD"].astype("string").str.strip()
    df["Fuente BD"] = df["Fuente BD"].fillna("Sin dato")

    # Código SGRH incrustado en "Codigo Fuente" -> permite vincular con el
    # proyecto "Libertad" (catastro Raigón), que usa ese mismo código.
    df["codigo_sgrh"] = df["Codigo Fuente"].astype("string").apply(_extraer_sgrh)

    df["Prof. Total"] = pd.to_numeric(df["Prof. Total"], errors="coerce")

    return df


def _extraer_sgrh(valor) -> str | None:
    if pd.isna(valor):
        return None
    match = _SGRH_RE.search(str(valor))
    return match.group(0).upper() if match else None


def _clasificar_por_latitud(df: pd.DataFrame, lat_col: str = "lat") -> pd.Series:
    resultado = pd.Series(pd.NA, index=df.index, dtype="object")
    con_lat = df[lat_col].notna()
    resultado.loc[con_lat & (df[lat_col] <= LATITUD_LIMITE_RAIGON_GUARANI)] = "Raigón"
    resultado.loc[con_lat & (df[lat_col] > LATITUD_LIMITE_RAIGON_GUARANI)] = "Guaraní"
    return resultado


def clasificar_acuifero_por_ubicacion(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon",
    limite_raigon=None, limite_guarani=None,
) -> pd.DataFrame:
    """
    Clasificación "oficial" de `acuifero`, en tres niveles de confianza
    (de mejor a peor):

    1. **Recorte por polígono real** (si se pasan `limite_raigon` /
       `limite_guarani`, geometrías shapely en EPSG:4326): el límite
       oficial del SAR (`LimiteCP`, ~2300 km², según De los Santos
       Gregoraschuk et al. 2019) y el área piloto de Tacuarembó. Es el
       criterio recomendado por el equipo (ver nota metodológica del
       proyecto) — la columna `Departamento` de la base PUK NO es
       confiable como filtro geográfico (mezcla mayúsculas/minúsculas,
       departamentos limítrofes, ~27 pozos con el departamento
       equivocado según sus propias coordenadas).
    2. **Latitud** (`_clasificar_por_latitud`): respaldo para pozos con
       coordenadas válidas pero ubicados fuera de ambos polígonos (p.ej.
       pozos reales en Rivera/Salto, fuera del área piloto acotada).
    3. **Departamento** (`acuifero_departamento`): último respaldo, solo
       para pozos sin coordenadas válidas.

    Requiere que el DataFrame ya tenga `lat`/`lon` (ver
    `geo_utils.utm_to_latlon` + `geo_utils.marcar_coordenadas_fuera_de_rango`)
    y `acuifero_departamento` (ver `normalizar_pozos`).
    """
    df = df.copy()

    por_poligono = pd.Series(pd.NA, index=df.index, dtype="object")
    metodo = pd.Series("departamento", index=df.index, dtype="object")

    if limite_raigon is not None and limite_guarani is not None:
        import geopandas as gpd

        con_coords = df[df[lat_col].notna() & df[lon_col].notna()]
        if not con_coords.empty:
            puntos = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(con_coords[lon_col], con_coords[lat_col]), crs="EPSG:4326"
            )
            dentro_raigon = puntos.within(limite_raigon).to_numpy()
            dentro_guarani = puntos.within(limite_guarani).to_numpy()
            por_poligono.loc[con_coords.index[dentro_raigon]] = "Raigón"
            por_poligono.loc[con_coords.index[dentro_guarani]] = "Guaraní"
            metodo.loc[con_coords.index[dentro_raigon | dentro_guarani]] = "poligono"

    por_latitud = _clasificar_por_latitud(df, lat_col)
    metodo.loc[por_poligono.isna() & por_latitud.notna()] = "latitud"

    df["acuifero"] = por_poligono.fillna(por_latitud).fillna(df["acuifero_departamento"])
    df["acuifero_metodo"] = metodo
    return df


def clasificar_formacion(formacion: str) -> str:
    if formacion in FORMACIONES_RAIGON:
        return "Raigón"
    if formacion in FORMACIONES_GUARANI:
        return "Guaraní"
    return "Otra/Sin clasificar"


def agregar_flags_de_cobertura(
    pozos: pd.DataFrame,
    hidraulica: pd.DataFrame,
    litologia: pd.DataFrame,
    hidroquimica: pd.DataFrame,
    historico_niveles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega columnas booleanas `tiene_hidraulica`, `tiene_litologia`,
    `tiene_hidroquimica`, `tiene_nivel_historico` y una columna numérica
    `n_fuentes_con_dato` (0 a 4) que resume la "completitud" del pozo.
    """
    df = pozos.copy()

    cod_hidraulica = set(hidraulica["Cod UK"].dropna())
    cod_litologia = set(litologia["Cod UK"].dropna())
    cod_hidroquimica = set(hidroquimica["Cod UK"].dropna())
    cod_historico = set(historico_niveles["Cod UK"].dropna())

    df["tiene_hidraulica"] = df["Cod UK"].isin(cod_hidraulica)
    df["tiene_litologia"] = df["Cod UK"].isin(cod_litologia)
    df["tiene_hidroquimica"] = df["Cod UK"].isin(cod_hidroquimica)
    df["tiene_nivel_historico"] = df["Cod UK"].isin(cod_historico)

    df["n_fuentes_con_dato"] = (
        df[["tiene_hidraulica", "tiene_litologia", "tiene_hidroquimica", "tiene_nivel_historico"]]
        .sum(axis=1)
        .astype(int)
    )
    return df


def formacion_principal_por_pozo(litologia: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada Cod UK, la formación más profunda relevada (aproximación
    de "formación de fondo del pozo", útil para el filtro de formaciones
    perforadas). Devuelve columnas: Cod UK, formacion_principal, grupo_formacion.
    """
    df = litologia.dropna(subset=["Formacion"]).copy()
    df["Fin"] = pd.to_numeric(df["Fin"], errors="coerce")
    df = df.dropna(subset=["Fin"])
    idx_mas_profunda = df.groupby("Cod UK")["Fin"].idxmax()
    principal = df.loc[idx_mas_profunda, ["Cod UK", "Formacion"]].rename(
        columns={"Formacion": "formacion_principal"}
    )
    principal["grupo_formacion"] = principal["formacion_principal"].apply(clasificar_formacion)
    return principal.reset_index(drop=True)
