"""
geo_utils.py
============
Conversión de coordenadas UTM 21S (EPSG:32721) a lat/lon (EPSG:4326).

A diferencia del proyecto "Libertad" (que usaba geopandas/shapely porque
había un polígono), acá todos los datos son puntos, así que usamos
`pyproj.Transformer` directamente sobre columnas de un DataFrame. Es
más liviano y más rápido de cargar en Streamlit Cloud.
"""
from __future__ import annotations

import pandas as pd
from pyproj import Transformer

CRS_UTM_21S = "EPSG:32721"
CRS_WGS84 = "EPSG:4326"

_transformer = Transformer.from_crs(CRS_UTM_21S, CRS_WGS84, always_xy=True)

# Bounding box aproximado de Uruguay (con margen). Coordenadas resultantes
# fuera de este rango son casi siempre errores de tipeo en la UTM de origen
# (dígitos de más/de menos), no ubicaciones reales.
LAT_MIN_URUGUAY, LAT_MAX_URUGUAY = -35.5, -29.5
LON_MIN_URUGUAY, LON_MAX_URUGUAY = -59.5, -52.5


def utm_to_latlon(
    df: pd.DataFrame,
    x_col: str = "X-UTM",
    y_col: str = "Y-UTM",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Agrega columnas `lat`/`lon` (WGS84) calculadas a partir de columnas
    UTM 21S. Filas con X/Y faltante o no numérico quedan con lat/lon NaN
    (no se descartan del DataFrame).
    """
    df = df.copy()
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    validos = x.notna() & y.notna()
    lon = pd.Series(index=df.index, dtype="float64")
    lat = pd.Series(index=df.index, dtype="float64")

    if validos.any():
        lon_v, lat_v = _transformer.transform(x[validos].to_numpy(), y[validos].to_numpy())
        lon.loc[validos] = lon_v
        lat.loc[validos] = lat_v

    df[lon_col] = lon
    df[lat_col] = lat
    return df


def marcar_coordenadas_fuera_de_rango(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
) -> pd.DataFrame:
    """
    Agrega `coordenada_valida` (bool) y pone en NaN el lat/lon de filas cuyo
    resultado cae fuera del bounding box de Uruguay (típicamente por un
    error de tipeo en la UTM original, ej. un dígito de más). Esas filas
    quedan fuera del mapa pero se conservan en las tablas/indicadores.
    """
    df = df.copy()
    dentro_de_rango = (
        df[lat_col].between(LAT_MIN_URUGUAY, LAT_MAX_URUGUAY)
        & df[lon_col].between(LON_MIN_URUGUAY, LON_MAX_URUGUAY)
    )
    df["coordenada_valida"] = dentro_de_rango.fillna(False)
    df.loc[~df["coordenada_valida"], [lat_col, lon_col]] = pd.NA
    return df
