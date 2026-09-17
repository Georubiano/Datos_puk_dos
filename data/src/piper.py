"""
piper.py
========
Diagrama de Piper (trilinear) a partir de la hoja "Hidroquímica". Requiere,
para cada muestra, los 4 cationes principales (Ca2+, Mg2+, Na+, K+) y los
3 aniones principales (Cl-, SO4 2-, HCO3-) medidos en la misma muestra.

Geometría: dos triángulos equiláteros (cationes a la izquierda, aniones a
la derecha) de lado 100, más el rombo central que combina ambos ejes
(Ca+Mg vs. Na+K, y HCO3 vs. SO4+Cl). El rombo se ubica directamente con
sus 4 vértices (izquierda = 100% Ca+Mg, derecha = 100% Na+K, abajo = 100%
HCO3, arriba = 100% SO4+Cl) y cada muestra se interpola linealmente entre
esos vértices — es la construcción estándar del diagrama de Piper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pesos equivalentes (peso atómico o molecular / valencia), para pasar de
# mg/L a meq/L.
_PESO_EQUIVALENTE = {
    "Ca": 20.039, "Mg": 12.1525, "Na": 22.98977, "K": 39.0983,
    "Cl": 35.453, "SO4": 48.03, "HCO3": 61.0168,
}

COLUMNAS_IONES = {
    "Ca": "Ca2+ (mg Ca2+/L)", "Mg": "Mg2+ (mg Mg2+/L)",
    "Na": "Na+ (mg Na+/L)", "K": "K+(mg K+/L)",
    "Cl": "Cl- (mg Cl-/L)", "SO4": "SO42- (mg SO42-/L)", "HCO3": "HCO3 mg/l",
}

# Vértices del triángulo de cationes (lado 100, apex hacia arriba).
_CAT_CA, _CAT_NAK, _CAT_MG = (0.0, 0.0), (100.0, 0.0), (50.0, 86.60254)
# Vértices del triángulo de aniones (offset a la derecha del de cationes;
# 30 unidades de separación para que las etiquetas de los vértices internos
# -Na++K+ del triángulo de cationes y HCO3- del de aniones- no se superpongan).
_AN_OFFSET = 130.0
_AN_HCO3, _AN_CL, _AN_SO4 = (_AN_OFFSET, 0.0), (_AN_OFFSET + 100.0, 0.0), (_AN_OFFSET + 50.0, 86.60254)
# Vértices del rombo central (izquierda=Ca+Mg, derecha=Na+K, abajo=HCO3, arriba=SO4+Cl),
# centrado en el punto medio entre ambos triángulos.
_CENTRO_X = (_CAT_NAK[0] + _AN_HCO3[0]) / 2
DIAMANTE_IZQ, DIAMANTE_DER = (_CENTRO_X - 50.0, 190.0), (_CENTRO_X + 50.0, 190.0)
DIAMANTE_ABAJO, DIAMANTE_ARRIBA = (_CENTRO_X, 103.39746), (_CENTRO_X, 276.60254)


def calcular_porcentajes_piper(hidroquimica: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra las muestras con los 7 iones principales medidos y calcula sus
    porcentajes en meq% (cationes y aniones suman 100 por separado).
    """
    columnas = list(COLUMNAS_IONES.values())
    df = hidroquimica.dropna(subset=columnas).copy()
    for ion, col in COLUMNAS_IONES.items():
        df[f"meq_{ion}"] = pd.to_numeric(df[col], errors="coerce") / _PESO_EQUIVALENTE[ion]
    df = df.dropna(subset=[f"meq_{ion}" for ion in COLUMNAS_IONES])

    suma_cationes = df["meq_Ca"] + df["meq_Mg"] + df["meq_Na"] + df["meq_K"]
    suma_aniones = df["meq_Cl"] + df["meq_SO4"] + df["meq_HCO3"]

    df["pct_Ca"] = 100 * df["meq_Ca"] / suma_cationes
    df["pct_Mg"] = 100 * df["meq_Mg"] / suma_cationes
    df["pct_NaK"] = 100 * (df["meq_Na"] + df["meq_K"]) / suma_cationes
    df["pct_Cl"] = 100 * df["meq_Cl"] / suma_aniones
    df["pct_SO4"] = 100 * df["meq_SO4"] / suma_aniones
    df["pct_HCO3"] = 100 * df["meq_HCO3"] / suma_aniones

    return df


def punto_cationes(pct_ca, pct_mg, pct_nak):
    x = pct_nak + 0.5 * pct_mg
    y = 0.8660254 * pct_mg
    return x, y


def punto_aniones(pct_hco3, pct_so4, pct_cl):
    x = _AN_OFFSET + pct_cl + 0.5 * pct_so4
    y = 0.8660254 * pct_so4
    return x, y


def punto_diamante(pct_nak, pct_so4_cl):
    """`pct_nak` = % Na+K entre cationes; `pct_so4_cl` = % (SO4+Cl) entre aniones."""
    x = DIAMANTE_IZQ[0] + pct_nak
    y = DIAMANTE_ABAJO[1] + 1.7320508 * pct_so4_cl
    return x, y


def agregar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x_cationes"], df["y_cationes"] = punto_cationes(df["pct_Ca"], df["pct_Mg"], df["pct_NaK"])
    df["x_aniones"], df["y_aniones"] = punto_aniones(df["pct_HCO3"], df["pct_SO4"], df["pct_Cl"])
    df["x_diamante"], df["y_diamante"] = punto_diamante(df["pct_NaK"], df["pct_SO4"] + df["pct_Cl"])
    return df


def dibujar_esqueleto(ax) -> None:
    """Dibuja los contornos de los dos triángulos y el rombo central, con etiquetas de vértice."""
    triangulo_cationes = plt_polygon([_CAT_CA, _CAT_NAK, _CAT_MG])
    triangulo_aniones = plt_polygon([_AN_HCO3, _AN_CL, _AN_SO4])
    rombo = plt_polygon([DIAMANTE_IZQ, DIAMANTE_ARRIBA, DIAMANTE_DER, DIAMANTE_ABAJO])

    for poligono in (triangulo_cationes, triangulo_aniones, rombo):
        ax.add_patch(poligono)

    etiquetas = [
        (_CAT_CA[0] - 3, _CAT_CA[1] - 6, "Ca²⁺", "right"),
        (_CAT_NAK[0] + 2, _CAT_NAK[1] - 6, "Na⁺+K⁺", "right"),
        (_CAT_MG[0], _CAT_MG[1] + 4, "Mg²⁺", "center"),
        (_AN_HCO3[0] - 2, _AN_HCO3[1] - 12, "HCO₃⁻", "left"),
        (_AN_CL[0] + 3, _AN_CL[1] - 6, "Cl⁻", "left"),
        (_AN_SO4[0], _AN_SO4[1] + 4, "SO₄²⁻", "center"),
        (DIAMANTE_IZQ[0] - 4, DIAMANTE_IZQ[1], "Ca²⁺+Mg²⁺", "right"),
        (DIAMANTE_DER[0] + 4, DIAMANTE_DER[1], "Na⁺+K⁺", "left"),
        (DIAMANTE_ABAJO[0], DIAMANTE_ABAJO[1] - 6, "HCO₃⁻", "center"),
        (DIAMANTE_ARRIBA[0], DIAMANTE_ARRIBA[1] + 4, "SO₄²⁻+Cl⁻", "center"),
    ]
    for x, y, texto, ha in etiquetas:
        ax.text(x, y, texto, ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-15, _AN_OFFSET + 115)
    ax.set_ylim(-15, DIAMANTE_ARRIBA[1] + 15)
    ax.set_aspect("equal")
    ax.axis("off")


def plt_polygon(vertices):
    from matplotlib.patches import Polygon
    return Polygon(vertices, closed=True, fill=False, edgecolor="black", linewidth=1.2)
