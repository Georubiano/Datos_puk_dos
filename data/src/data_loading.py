"""
data_loading.py
================
Lectura de la base consolidada de pozos (Sistema Acuífero Raigón +
Sistema Acuífero Guaraní), archivo `base_final.xlsx` con 9 hojas
relacionadas por el identificador único `Cod UK`.

Devuelve DataFrames "crudos" (sin limpiar todavía) — la limpieza y
clasificación por acuífero vive en `cleaning.py`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BASE_FINAL_PATH = RAW_DIR / "base_final.xlsx"

LIMITE_RAIGON_PATH = RAW_DIR / "limites" / "limite_cp" / "LimiteCP.shp"
LIMITE_GUARANI_PATH = RAW_DIR / "limites" / "area_piloto_tacuarembo" / "Area_Piloto_Tacuarembo_entera.shp"

HOJAS = {
    "pozos": "Pozos",
    "hidraulica": "Hidráulica",
    "napas": "Ubic. Napas_m",
    "filtros": "Filtros_m",
    "litologia": "Litología",
    "hidroquimica": "Hidroquímica",
    "historico_niveles": "Historico niveles",
    "formacion": "Formación",
    "fuente": "Fuente",
}


def load_all_raw(path: Path | str = BASE_FINAL_PATH) -> dict[str, pd.DataFrame]:
    """Lee las 9 hojas del Excel y las devuelve en un dict por nombre corto."""
    datos = {}
    for clave, hoja in HOJAS.items():
        datos[clave] = pd.read_excel(path, sheet_name=hoja, engine="openpyxl")
    return datos


def load_limites_acuiferos(
    ruta_raigon: Path | str = LIMITE_RAIGON_PATH, ruta_guarani: Path | str = LIMITE_GUARANI_PATH
):
    """
    Carga los polígonos de límite del SAR (Raigón) y del área piloto de
    Tacuarembó (Guaraní), usados para clasificar el acuífero de cada pozo
    por ubicación real (ver `cleaning.clasificar_acuifero_por_ubicacion`).
    Requiere `geopandas` (import local para no exigirlo si esta función
    no se usa). Devuelve las dos geometrías disueltas, en EPSG:4326.
    """
    import geopandas as gpd

    limite_raigon = gpd.read_file(ruta_raigon).to_crs("EPSG:4326").union_all()
    limite_guarani = gpd.read_file(ruta_guarani).to_crs("EPSG:4326").union_all()
    return limite_raigon, limite_guarani


if __name__ == "__main__":
    datos = load_all_raw()
    for nombre, df in datos.items():
        print(f"{nombre}: {df.shape}")
