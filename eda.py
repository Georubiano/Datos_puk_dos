"""
eda.py
======
Funciones que arman los indicadores (KPIs y tablas) que se muestran en
la app: cantidad de pozos por acuífero, por fuente, con geoquímica,
con litología, cobertura temporal, etc.

Todas reciben el DataFrame `pozos` ya normalizado y con los flags de
cobertura agregados (ver `cleaning.agregar_flags_de_cobertura`).
"""
from __future__ import annotations

import pandas as pd


def pozos_por_acuifero(pozos: pd.DataFrame) -> pd.DataFrame:
    return (
        pozos.groupby("acuifero", dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
        .sort_values("cantidad_pozos", ascending=False)
    )


def pozos_por_acuifero_y_fuente(pozos: pd.DataFrame) -> pd.DataFrame:
    return (
        pozos.groupby(["acuifero", "Fuente BD"], dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
        .sort_values(["acuifero", "cantidad_pozos"], ascending=[True, False])
    )


def cobertura_por_acuifero(pozos: pd.DataFrame) -> pd.DataFrame:
    """Cantidad y % de pozos con hidráulica / litología / hidroquímica / histórico, por acuífero."""
    columnas = ["tiene_hidraulica", "tiene_litologia", "tiene_hidroquimica", "tiene_nivel_historico"]
    resumen = pozos.groupby("acuifero", dropna=False)[columnas].sum().reset_index()
    totales = pozos.groupby("acuifero", dropna=False).size().rename("total_pozos")
    resumen = resumen.merge(totales, on="acuifero")
    for col in columnas:
        resumen[f"{col}_pct"] = (100 * resumen[col] / resumen["total_pozos"]).round(1)
    return resumen


def cobertura_por_fuente_detallada(pozos: pd.DataFrame) -> pd.DataFrame:
    """
    Cantidad de pozos con cada tipo de dato relevado (hidráulica, litología,
    hidroquímica, serie histórica), abierto por acuífero y por Fuente BD.
    Formato largo: columnas acuifero, tipo_dato, Fuente BD, cantidad_pozos.
    Pensado para un gráfico de barras apiladas por Fuente BD dentro de
    cada tipo de dato.
    """
    columnas = {
        "tiene_hidraulica": "Hidráulica", "tiene_litologia": "Litología",
        "tiene_hidroquimica": "Hidroquímica", "tiene_nivel_historico": "Serie histórica",
    }
    largo = pozos.melt(
        id_vars=["acuifero", "Fuente BD"], value_vars=list(columnas.keys()),
        var_name="flag", value_name="tiene_dato",
    )
    largo = largo[largo["tiene_dato"]].copy()
    largo["tipo_dato"] = largo["flag"].map(columnas)
    return (
        largo.groupby(["acuifero", "tipo_dato", "Fuente BD"], dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
    )


def cobertura_por_acuifero_y_fuente(pozos: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """
    Cantidad de pozos con `flag_col in {True}` (p. ej. tiene_hidroquimica),
    agrupado por acuífero y Fuente BD.
    """
    df = pozos[pozos[flag_col]]
    return (
        df.groupby(["acuifero", "Fuente BD"], dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
        .sort_values(["acuifero", "cantidad_pozos"], ascending=[True, False])
    )


def resumen_completitud(pozos: pd.DataFrame) -> pd.DataFrame:
    """Distribución de pozos según cuántas fuentes de datos adicionales tienen (0 a 4)."""
    return (
        pozos.groupby(["acuifero", "n_fuentes_con_dato"], dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
        .sort_values(["acuifero", "n_fuentes_con_dato"])
    )


def resumen_profundidad(pozos: pd.DataFrame) -> pd.DataFrame:
    return (
        pozos.groupby("acuifero", dropna=False)["Prof. Total"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(1)
        .reset_index()
    )


def formaciones_por_acuifero(pozos_con_formacion: pd.DataFrame) -> pd.DataFrame:
    """`pozos_con_formacion` = pozos ya mergeado con formacion_principal_por_pozo()."""
    return (
        pozos_con_formacion.dropna(subset=["formacion_principal"])
        .groupby(["acuifero", "formacion_principal"], dropna=False)
        .size()
        .reset_index(name="cantidad_pozos")
        .sort_values(["acuifero", "cantidad_pozos"], ascending=[True, False])
    )


def parametros_hidroquimicos_mas_medidos(hidroquimica: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Ranking de columnas de parámetros (no metadata) por cantidad de valores no nulos."""
    columnas_metadata = {"Cod UK", "Fecha", "Año", "Observaciones", "Fuente", "Fuente.1"}
    columnas_parametros = [c for c in hidroquimica.columns if c not in columnas_metadata]
    conteos = hidroquimica[columnas_parametros].notna().sum().sort_values(ascending=False)
    return conteos.head(top_n).reset_index().rename(columns={"index": "parametro", 0: "cantidad_mediciones"})
