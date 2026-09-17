"""
streamlit_app.py
================
App de visualización — Sistema Acuífero Raigón + Sistema Acuífero Guaraní
(DINAGUA). Selector de acuífero, filtros por fuente de datos y por tipo
de información relevada (litología / hidroquímica / hidráulica / histórico
de niveles), y modo claro/oscuro (aplicado también al mapa PyDeck).

Estilo de referencia: panel "Gestión de Recursos Hídricos - DINAGUA"
(encabezados con emoji + separadores, fila de métricas, gráficos
Matplotlib/Seaborn, mapa PyDeck sobre basemap Carto).

Ejecutar con:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import seaborn as sns
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cleaning as cl  # noqa: E402
import data_loading as dl  # noqa: E402
import eda  # noqa: E402
import geo_utils as gu  # noqa: E402
import series_temporales as serie_mod  # noqa: E402

st.set_page_config(page_title="Pozos - Raigón / Guaraní", page_icon="💧", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Carga y preparación de datos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    datos = dl.load_all_raw()

    pozos = cl.normalizar_pozos(datos["pozos"])
    pozos = cl.agregar_flags_de_cobertura(
        pozos, datos["hidraulica"], datos["litologia"], datos["hidroquimica"], datos["historico_niveles"]
    )
    principal = cl.formacion_principal_por_pozo(datos["litologia"])
    pozos = pozos.merge(principal, on="Cod UK", how="left")

    pozos = gu.utm_to_latlon(pozos)
    pozos = gu.marcar_coordenadas_fuera_de_rango(pozos)
    # Clasificación "oficial" de acuífero por ubicación real: recorte por
    # el límite geométrico del SAR / área piloto Tacuarembó (mejor criterio,
    # recomendado por el equipo), con latitud y Departamento como respaldo
    # para pozos fuera de esos polígonos o sin coordenadas — ver README.
    limite_raigon, limite_guarani = dl.load_limites_acuiferos()
    pozos = cl.clasificar_acuifero_por_ubicacion(
        pozos, limite_raigon=limite_raigon, limite_guarani=limite_guarani
    )

    return {
        "pozos": pozos,
        "hidraulica": datos["hidraulica"],
        "litologia": datos["litologia"],
        "hidroquimica": datos["hidroquimica"],
        "historico_niveles": datos["historico_niveles"],
    }


@st.cache_data
def cargar_serie_temporal_niveles():
    return serie_mod.construir_serie_temporal_niveles()


datos = cargar_datos()
pozos = datos["pozos"]
serie_niveles = cargar_serie_temporal_niveles()

# Última lectura hidráulica (NE, ND, Caudal) por pozo — se usa tanto en la
# ficha de Litología interactiva como en la pestaña de Hidráulica.
hidraulica_cruda = datos["hidraulica"].copy()
hidraulica_cruda["NE_m"] = pd.to_numeric(hidraulica_cruda["NE_m"], errors="coerce")
hidraulica_cruda["ND-m"] = pd.to_numeric(hidraulica_cruda["ND-m"], errors="coerce")
hidraulica_ultima = (
    hidraulica_cruda.sort_values("Año", ascending=False)
    .groupby("Cod UK", as_index=False)
    .first()
)

# ---------------------------------------------------------------------------
# Sidebar: apariencia + filtros
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Apariencia")
modo_oscuro = st.sidebar.toggle("🌙 Modo oscuro", value=False)
st.sidebar.markdown("---")

st.sidebar.markdown("## Filtros")
acuiferos_disponibles = ["Raigón", "Guaraní", "Sin clasificar"]
acuifero_sel = st.sidebar.multiselect(
    "Acuífero", options=acuiferos_disponibles, default=["Raigón", "Guaraní"]
)

fuentes_disponibles = sorted(pozos["Fuente BD"].dropna().unique().tolist())
fuentes_sel = st.sidebar.multiselect("Fuente BD", options=fuentes_disponibles, default=fuentes_disponibles)

_departamento_serie = pozos["Departamento"].fillna("Sin dato")
departamentos_disponibles = sorted(_departamento_serie.unique().tolist())
departamento_sel = st.sidebar.multiselect(
    "Departamento", options=departamentos_disponibles, default=departamentos_disponibles
)

st.sidebar.markdown("**Con datos relevados en:**")
solo_litologia = st.sidebar.checkbox("Litología")
solo_hidroquimica = st.sidebar.checkbox("Hidroquímica")
solo_hidraulica = st.sidebar.checkbox("Hidráulica (caudal / niveles)")
solo_historico = st.sidebar.checkbox("Serie histórica de niveles")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Fuente: [DINAGUA](https://www.gub.uy/ministerio-ambiente/) "
    "— Ministerio de Ambiente, Uruguay"
)

pozos_f = pozos[
    pozos["acuifero"].isin(acuifero_sel)
    & pozos["Fuente BD"].isin(fuentes_sel)
    & _departamento_serie.isin(departamento_sel)
]
if solo_litologia:
    pozos_f = pozos_f[pozos_f["tiene_litologia"]]
if solo_hidroquimica:
    pozos_f = pozos_f[pozos_f["tiene_hidroquimica"]]
if solo_hidraulica:
    pozos_f = pozos_f[pozos_f["tiene_hidraulica"]]
if solo_historico:
    pozos_f = pozos_f[pozos_f["tiene_nivel_historico"]]

# ---------------------------------------------------------------------------
# Tema: CSS + paleta según modo claro/oscuro
# ---------------------------------------------------------------------------
if modo_oscuro:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #f0f2f6; }
        section[data-testid="stSidebar"] { background-color: #161a23; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    sns.set_theme(style="darkgrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
        "savefig.facecolor": "#0e1117", "text.color": "#f0f2f6",
        "axes.labelcolor": "#f0f2f6", "xtick.color": "#f0f2f6", "ytick.color": "#f0f2f6",
    })
    # Basemap vectorial CartoDB "Dark Matter" (gratuito, sin API key).
    MAPA_ESTILO = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
else:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set_theme(style="whitegrid", context="notebook")
    # Basemap vectorial CartoDB "Positron" (gratuito, sin API key).
    MAPA_ESTILO = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# Colores del ciclo por defecto de Matplotlib ("C0".."C9" = tab10), resueltos
# a hex para poder reutilizarlos también en Plotly/PyDeck. Raigón = C1,
# Guaraní = C9, tal como se pidió.
_CICLO_MPL = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_ACUIFERO = {"Raigón": _CICLO_MPL[1], "Guaraní": _CICLO_MPL[9], "Sin clasificar": _CICLO_MPL[7]}

# Paleta institucional armonizada (tonos categóricos estilo Tableau/ColorBrewer),
# una entrada fija por Fuente BD. Se usa tanto en el mapa como en los gráficos
# de cobertura, para que el color de cada fuente sea siempre el mismo en toda
# la app.
COLOR_FUENTE = {
    "Dinagua": "#0284C7",   # Azul oceánico — público / agua
    "OSE": "#0D9488",       # Verde azulado / teal — servicios / agua potable
    "Ancap": "#D97706",     # Ámbar / naranja — energía / industria
    "Dinamige": "#4F46E5",  # Índigo — minería / geología
    "MEVIR": "#059669",     # Verde esmeralda — vivienda / social
    "Privado": "#8B5CF6",   # Púrpura suave
    "Academia": "#DB2777",  # Rosa/frambuesa — investigación
    "Sin dato": "#94A3B8",  # Gris neutro
}

# Paleta "geológica" de tonos tierra (ámbar, arena, gris pizarra y variaciones),
# usada para colorear formaciones en la columna litológica — más apropiada
# para un gráfico técnico de estratigrafía que una paleta cualitativa brillante.
PALETA_GEOLOGICA = [
    "#D97706", "#FBBF24", "#64748B", "#92400E", "#A16207", "#78716C",
    "#B45309", "#CA8A04", "#57534E", "#EA580C", "#854D0E", "#A8A29E",
    "#7C2D12", "#D6D3D1", "#451A03",
]

# Límites de referencia para agua de consumo (Decreto 253/79 y normativa UNIT
# vigente en Uruguay). Solo se marcan los parámetros con un límite simple y
# bien definido; "pH" se trata aparte por tener rango (mín/máx), no un único
# umbral. Claves = nombre EXACTO de columna en la hoja de Hidroquímica.
UMBRALES_NORMATIVOS = {
    "As (ug/L)": 10.0,
    "As disuelto(ug/L)": 10.0,
    "F- (mg F-/L)": 1.3,
    "Fe total (mg/L)": 0.3,
    "Fe disuelto (mg/L)": 0.3,
    "Mn (mg/L)": 0.1,
    "Mn disuelto (mg/L)": 0.1,
    "NO3 (mg/L)": 50.0,
    "Conductividad electrica (microS/cm)": 2500.0,
}
UMBRAL_PH = (6.5, 8.5)


def _hex_a_rgb(color_hex: str) -> list[int]:
    color_hex = color_hex.lstrip("#")
    return [int(color_hex[i:i + 2], 16) for i in (0, 2, 4)]


def _limpiar_texto(texto: str) -> str:
    """Reemplaza guiones bajos por espacios en labels/ticks (ej. nombres de columna)."""
    return texto.replace("_", " ") if isinstance(texto, str) else texto


def _estilizar_ejes(ax) -> None:
    """
    Estilo común para los gráficos de barra: ejes x/y en negro, con ticks
    mayores (con etiqueta) y menores (sin etiqueta) marcados, y sin
    guiones bajos en los textos de los ejes.
    """
    sns.despine(ax=ax)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color((0, 0, 0))
        ax.spines[lado].set_linewidth(1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", color="black", labelsize=9)
    ax.tick_params(axis="both", which="minor", color="black", length=2.5)

    ax.set_xlabel(_limpiar_texto(ax.get_xlabel()))
    ax.set_ylabel(_limpiar_texto(ax.get_ylabel()))
    ax.set_xticklabels([_limpiar_texto(t.get_text()) for t in ax.get_xticklabels()])
    ax.set_yticklabels([_limpiar_texto(t.get_text()) for t in ax.get_yticklabels()])
    leyenda = ax.get_legend()
    if leyenda is not None:
        leyenda.set_title(_limpiar_texto(leyenda.get_title().get_text()))


def _para_mostrar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copia "segura para mostrar" de un DataFrame crudo: convierte a texto
    las columnas de tipo object con contenido mixto (ej. la columna
    "Inicio" de Litología, que mezcla números con el texto "Falta"), que
    si no rompen la conversión a tabla de Streamlit (pyarrow.ArrowInvalid).
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": "", "NaT": ""})
    return df


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("💧 Sistema Acuífero Raigón y Sistema Acuífero Guaraní — DINAGUA")
st.markdown(
    "Panel interactivo de pozos y niveles de agua subterránea del estudio piloto "
    "Raigón (San José) y Guaraní (Tacuarembó)."
)
st.markdown("---")

tab_resumen, tab_litologia, tab_hidroquimica, tab_hidraulica, tab_dj = st.tabs(
    [
        "📋 Resumen general", "🪨 Litología interactiva", "🧪 Hidroquímica interactiva",
        "💧 Hidráulica", "📝 Declaraciones Juradas",
    ]
)

with tab_resumen:
    # ---------------------------------------------------------------------------
    # Sección 1 — Panel inicial (métricas)
    # ---------------------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pozos (filtro actual)", f"{len(pozos_f):,}")
    c2.metric("Con litología", f"{int(pozos_f['tiene_litologia'].sum()):,}")
    c3.metric("Con hidroquímica", f"{int(pozos_f['tiene_hidroquimica'].sum()):,}")
    c4.metric("Con serie histórica", f"{int(pozos_f['tiene_nivel_historico'].sum()):,}")
    c5.metric("Acuíferos representados", f"{pozos_f['acuifero'].nunique()}")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 2 — Mapa de pozos relevados (PyDeck)
    # ---------------------------------------------------------------------------
    st.header("🗺️ Mapa de Pozos Relevados")

    con_coords = pozos_f.dropna(subset=["lat", "lon"]).copy()

    if con_coords.empty:
        st.warning("No hay pozos con coordenadas válidas para el filtro seleccionado.")
    else:
        con_coords["color"] = con_coords["Fuente BD"].apply(
            lambda f: _hex_a_rgb(COLOR_FUENTE.get(f, "#999999")) + [179]  # ~70% opacidad
        )
        mapa_df = con_coords.rename(columns={
            "Cod UK": "cod_uk", "Fuente BD": "fuente_bd", "acuifero": "acuifero_pozo",
        })[["lat", "lon", "color", "cod_uk", "acuifero_pozo", "fuente_bd", "tiene_litologia", "tiene_hidroquimica"]]

        fuentes_presentes = sorted(con_coords["Fuente BD"].dropna().unique().tolist())
        chips_html = ""
        for fuente in fuentes_presentes:
            hex_c = COLOR_FUENTE.get(fuente, "#999999")
            chips_html += (
                f"<span style='background:{hex_c};color:white;border:1px solid rgba(0,0,0,0.25);"
                f"padding:4px 12px;border-radius:14px;font-size:12px;font-weight:600;'>{fuente}</span>"
            )
        fondo_tarjeta = "#161a23" if modo_oscuro else "#ffffff"
        borde_tarjeta = "#2a2f3a" if modo_oscuro else "#e2e8f0"
        color_texto_tarjeta = "#f0f2f6" if modo_oscuro else "#334155"
        leyenda_html = (
            f"<div style='background:{fondo_tarjeta};border:1px solid {borde_tarjeta};"
            "border-radius:10px;padding:10px 14px;margin-bottom:12px;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.12);'>"
            f"<div style='font-size:12px;font-weight:700;color:{color_texto_tarjeta};margin-bottom:6px;'>"
            "Leyenda — color por Fuente BD</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;'>{chips_html}</div></div>"
        )
        st.markdown(leyenda_html, unsafe_allow_html=True)
        st.caption(f"📍 {len(mapa_df):,} pozos georreferenciados (puntos con borde fino, color = Fuente BD).")

        capa = pdk.Layer(
            "ScatterplotLayer", data=mapa_df,
            get_position=["lon", "lat"], get_fill_color="color",
            get_line_color=[0, 0, 0, 150], line_width_min_pixels=0.5,
            stroked=True, get_radius=800, pickable=True, auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[capa],
            initial_view_state=pdk.ViewState(
                latitude=float(con_coords["lat"].mean()), longitude=float(con_coords["lon"].mean()), zoom=6.5
            ),
            map_style=MAPA_ESTILO,
            tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
        ))

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 3 — Indicadores por acuífero
    # ---------------------------------------------------------------------------
    st.header("📊 Indicadores por Acuífero")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Pozos por acuífero**")
        resumen_ac = eda.pozos_por_acuifero(pozos_f)
        total_ac = resumen_ac["cantidad_pozos"].sum()
        fig_pie = px.pie(
            resumen_ac, values="cantidad_pozos", names="acuifero", color="acuifero",
            color_discrete_map=COLOR_ACUIFERO, hole=0.35,
        )
        fig_pie.update_traces(
            textinfo="label+value",
            hovertemplate="%{label}: %{percent} (%{value} pozos)<extra></extra>",
        )
        fig_pie.update_layout(
            title=dict(text="Pozos por acuífero", font=dict(size=16)),
            template="plotly_dark" if modo_oscuro else "plotly_white",
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(f"Total: {total_ac:,} pozos. Pasá el mouse sobre cada porción para ver el porcentaje.")

    with col_b:
        st.markdown("**Ranking de Fuente BD, por acuífero**")
        resumen_fuente = eda.pozos_por_acuifero_y_fuente(pozos_f)
        orden_fuente = (
            resumen_fuente.groupby("Fuente BD")["cantidad_pozos"].sum().sort_values(ascending=True).index
        )
        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.barplot(
            data=resumen_fuente, y="Fuente BD", x="cantidad_pozos", hue="acuifero",
            order=orden_fuente, palette=COLOR_ACUIFERO, ax=ax,
        )
        ax.set_title("Pozos por Fuente BD y acuífero", fontweight="bold")
        ax.set_xlabel("Cantidad de pozos")
        ax.set_ylabel("")
        ax.legend(title="Acuífero", fontsize=9)
        _estilizar_ejes(ax)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**Cobertura de datos por acuífero, diferenciada por Fuente BD**")
    cobertura_detalle = eda.cobertura_por_fuente_detallada(pozos_f)
    orden_tipo_dato = ["Hidráulica", "Litología", "Hidroquímica", "Serie histórica"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    acuiferos_presentes = [a for a in ["Raigón", "Guaraní"] if a in cobertura_detalle["acuifero"].unique()]
    for ax, acuifero_nombre in zip(axes, acuiferos_presentes or ["Raigón", "Guaraní"]):
        sub = cobertura_detalle[cobertura_detalle["acuifero"] == acuifero_nombre]
        pivote = (
            sub.pivot_table(index="tipo_dato", columns="Fuente BD", values="cantidad_pozos", fill_value=0)
            .reindex(orden_tipo_dato, fill_value=0)
        )
        abajo = np.zeros(len(pivote))
        for fuente in pivote.columns:
            valores = pivote[fuente].to_numpy()
            ax.bar(pivote.index, valores, bottom=abajo, color=COLOR_FUENTE.get(fuente, "#999999"),
                   edgecolor="black", linewidth=0.5, label=fuente)
            abajo += valores
        ax.set_title(acuifero_nombre, fontweight="bold", color=COLOR_ACUIFERO.get(acuifero_nombre, "black"))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        _estilizar_ejes(ax)

    axes[0].set_ylabel("Cantidad de pozos")
    handles, labels = axes[-1].get_legend_handles_labels()
    por_fuente = dict(zip(labels, handles))
    fig.legend(por_fuente.values(), por_fuente.keys(), title="Fuente BD", loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=min(len(por_fuente), 8), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Distribución de profundidad total por acuífero"):
        st.dataframe(eda.resumen_profundidad(pozos_f), use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 4 — Litología
    # ---------------------------------------------------------------------------
    st.header("🪨 Litología y Formaciones")

    pozos_con_formacion = pozos_f.dropna(subset=["formacion_principal"])
    if pozos_con_formacion.empty:
        st.info("No hay pozos con litología relevada para el filtro seleccionado.")
    else:
        formaciones = eda.formaciones_por_acuifero(pozos_con_formacion)
        orden_formacion = (
            formaciones.groupby("formacion_principal")["cantidad_pozos"].sum().sort_values(ascending=False).index
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=formaciones, x="formacion_principal", y="cantidad_pozos", hue="acuifero",
            order=orden_formacion, palette=COLOR_ACUIFERO, ax=ax,
        )
        ax.set_title("Formación geológica principal (más profunda relevada) por acuífero", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Cantidad de pozos")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Acuífero", fontsize=9)
        _estilizar_ejes(ax)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Ver tabla cruda de Litología"):
        st.dataframe(_para_mostrar(datos["litologia"]), use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 5 — Hidroquímica
    # ---------------------------------------------------------------------------
    st.header("🧪 Hidroquímica")

    st.markdown("**Parámetros con más mediciones** (toda la base, sin filtrar por acuífero)")
    top_n = st.slider("Cantidad de parámetros a mostrar", 5, 30, 15)
    ranking = eda.parametros_hidroquimicos_mas_medidos(datos["hidroquimica"], top_n=top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.32)))
    sns.barplot(data=ranking, x="cantidad_mediciones", y="parametro", hue="parametro", palette="tab10", legend=False, ax=ax)
    ax.set_title("Parámetros hidroquímicos con más mediciones", fontweight="bold")
    ax.set_xlabel("Cantidad de mediciones")
    ax.set_ylabel("")
    _estilizar_ejes(ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Pozos con hidroquímica, por acuífero y Fuente BD**")
    st.dataframe(eda.cobertura_por_acuifero_y_fuente(pozos_f, "tiene_hidroquimica"), use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 6 — Series temporales de niveles
    # ---------------------------------------------------------------------------
    st.header("📈 Series Temporales de Niveles de Agua")

    cod_uk_con_serie = set(serie_niveles["Cod UK"].unique())
    pozos_con_serie = pozos_f[
        pozos_f["Cod UK"].isin(cod_uk_con_serie) & pozos_f["lat"].notna() & pozos_f["lon"].notna()
    ]

    if pozos_con_serie.empty:
        st.info("No hay pozos con serie histórica de niveles para el filtro seleccionado.")
    else:
        opciones_pozo = sorted(pozos_con_serie["Cod UK"].unique().tolist())
        pozo_sel = st.selectbox("Elegí un pozo para ver su serie temporal", options=opciones_pozo)

        mapa_serie = pozos_con_serie.rename(
            columns={"Cod UK": "cod_uk", "acuifero": "acuifero_pozo", "Fuente BD": "fuente_bd"}
        ).copy()
        mapa_serie["es_seleccionado"] = mapa_serie["cod_uk"] == pozo_sel
        mapa_serie["color"] = mapa_serie["es_seleccionado"].apply(
            lambda sel: [230, 25, 25, 255] if sel else [120, 120, 120, 140]
        )
        mapa_serie["radio"] = mapa_serie["es_seleccionado"].apply(lambda sel: 1800 if sel else 700)
        fila_sel = mapa_serie[mapa_serie["es_seleccionado"]].iloc[0]

        col_mapa, col_serie = st.columns([1, 1.3])

        with col_mapa:
            st.markdown("**Ubicación de pozos con serie histórica**")
            capa_serie = pdk.Layer(
                "ScatterplotLayer", data=mapa_serie,
                get_position=["lon", "lat"], get_fill_color="color",
                get_line_color=[0, 0, 0, 180], line_width_min_pixels=0.6,
                stroked=True, get_radius="radio", pickable=True, auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[capa_serie],
                initial_view_state=pdk.ViewState(
                    latitude=float(fila_sel["lat"]), longitude=float(fila_sel["lon"]), zoom=9,
                ),
                map_style=MAPA_ESTILO,
                tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
            ))
            st.caption("🔴 Pozo seleccionado. ⚪ Resto de los pozos con serie histórica (filtro actual).")

        with col_serie:
            st.markdown(f"**Serie de niveles — {pozo_sel}**")
            serie_pozo = serie_niveles[serie_niveles["Cod UK"] == pozo_sel].sort_values("fecha")
            fig, ax = plt.subplots(figsize=(7, 4.6))
            color_pozo = COLOR_ACUIFERO.get(fila_sel["acuifero_pozo"], "#377eb8")
            ax.plot(
                serie_pozo["fecha"], serie_pozo["nivel"], marker="o", markersize=4,
                linewidth=1.3, color=color_pozo,
            )
            ax.set_title(f"Nivel de agua registrado — {pozo_sel}", fontweight="bold")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Nivel")
            fig.autofmt_xdate()
            _estilizar_ejes(ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(
                f"{len(serie_pozo)} mediciones entre {serie_pozo['fecha'].min():%m/%Y} y "
                f"{serie_pozo['fecha'].max():%m/%Y}. Unidad de nivel según fuente original del dato."
            )

        with st.expander("Ver tabla de la serie seleccionada"):
            st.dataframe(
                serie_pozo.rename(columns={"fecha": "Fecha", "nivel": "Nivel"}), use_container_width=True
            )

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Sección 7 — Datos
    # ---------------------------------------------------------------------------
    st.header("📄 Datos")

    columnas_mostrar = [
        "Cod UK", "acuifero", "acuifero_metodo", "Departamento", "Fuente BD", "Localidad",
        "Prof. Total", "formacion_principal", "tiene_hidraulica", "tiene_litologia",
        "tiene_hidroquimica", "tiene_nivel_historico", "n_fuentes_con_dato", "codigo_sgrh",
    ]
    st.dataframe(pozos_f[columnas_mostrar], use_container_width=True)
    st.download_button(
        "Descargar como CSV",
        data=pozos_f[columnas_mostrar].to_csv(index=False).encode("utf-8"),
        file_name="pozos_raigon_guarani_filtrado.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 2 — Litología interactiva: mapa + perfil por pozo
# ---------------------------------------------------------------------------
with tab_litologia:
    st.header("🪨 Litología interactiva — Mapa y perfil por pozo")
    st.markdown(
        "Elegí un pozo (en la lista) para ver su perfil litológico: cada barra "
        "representa una formación geológica, desde la profundidad de inicio hasta la de fin."
    )

    litologia_cruda = datos["litologia"].copy()
    litologia_cruda["Inicio"] = pd.to_numeric(litologia_cruda["Inicio"], errors="coerce")
    litologia_cruda["Fin"] = pd.to_numeric(litologia_cruda["Fin"], errors="coerce")
    litologia_valida = litologia_cruda.dropna(subset=["Inicio", "Fin"])

    cod_uk_con_litologia = set(litologia_valida["Cod UK"].unique())
    pozos_con_litologia = pozos_f[
        pozos_f["Cod UK"].isin(cod_uk_con_litologia) & pozos_f["lat"].notna() & pozos_f["lon"].notna()
    ]

    if pozos_con_litologia.empty:
        st.info("No hay pozos con litología (con profundidades numéricas) para el filtro seleccionado.")
    else:
        opciones_litologia = sorted(pozos_con_litologia["Cod UK"].unique().tolist())
        pozo_lito_sel = st.selectbox(
            "Elegí un pozo con litología relevada", options=opciones_litologia, key="pozo_lito_sel"
        )

        mapa_lito = pozos_con_litologia.rename(
            columns={"Cod UK": "cod_uk", "acuifero": "acuifero_pozo", "Fuente BD": "fuente_bd"}
        ).copy()
        mapa_lito["es_seleccionado"] = mapa_lito["cod_uk"] == pozo_lito_sel
        mapa_lito["color"] = mapa_lito["es_seleccionado"].apply(
            lambda sel: [230, 25, 25, 255] if sel else [50, 120, 50, 140]
        )
        mapa_lito["radio"] = mapa_lito["es_seleccionado"].apply(lambda sel: 1800 if sel else 700)
        fila_lito_sel = mapa_lito[mapa_lito["es_seleccionado"]].iloc[0]

        _perfil_pozo_sel = litologia_valida[litologia_valida["Cod UK"] == pozo_lito_sel]
        _prof_total = _perfil_pozo_sel["Fin"].max() if not _perfil_pozo_sel.empty else None
        _hid_pozo_sel = hidraulica_ultima[hidraulica_ultima["Cod UK"] == pozo_lito_sel]
        _ne_pozo_sel = _hid_pozo_sel["NE_m"].iloc[0] if not _hid_pozo_sel.empty else None
        _caudal_pozo_sel = _hid_pozo_sel["Caudal"].iloc[0] if (
            not _hid_pozo_sel.empty and "Caudal" in _hid_pozo_sel.columns
        ) else None

        with st.container(border=True):
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            col_f1.metric("Profundidad total", f"{_prof_total:.1f} m" if pd.notna(_prof_total) else "Sin dato")
            col_f2.metric("Nivel estático (NE)", f"{_ne_pozo_sel:.2f} m" if pd.notna(_ne_pozo_sel) else "Sin dato")
            col_f3.metric("Acuífero captado", str(fila_lito_sel["acuifero_pozo"]))
            col_f4.metric(
                "Caudal", f"{_caudal_pozo_sel:.1f} m³/h" if pd.notna(_caudal_pozo_sel) else "Sin dato"
            )

        col_mapa_lito, col_perfil = st.columns([1, 1.2])

        with col_mapa_lito:
            st.markdown("**Pozos con litología relevada**")
            capa_lito = pdk.Layer(
                "ScatterplotLayer", data=mapa_lito,
                get_position=["lon", "lat"], get_fill_color="color",
                get_line_color=[0, 0, 0, 180], line_width_min_pixels=0.6,
                stroked=True, get_radius="radio", pickable=True, auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[capa_lito],
                initial_view_state=pdk.ViewState(
                    latitude=float(fila_lito_sel["lat"]), longitude=float(fila_lito_sel["lon"]), zoom=9,
                ),
                map_style=MAPA_ESTILO,
                tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
            ))
            st.caption("🔴 Pozo seleccionado. 🟢 Resto de los pozos con litología (filtro actual).")

        with col_perfil:
            st.markdown(f"**Perfil litológico — {pozo_lito_sel}**")
            perfil = litologia_valida[litologia_valida["Cod UK"] == pozo_lito_sel].sort_values("Inicio")

            formaciones_todas = sorted(litologia_valida["Formacion"].dropna().unique().tolist())
            color_formacion = {
                f: PALETA_GEOLOGICA[i % len(PALETA_GEOLOGICA)] for i, f in enumerate(formaciones_todas)
            }

            fig, ax = plt.subplots(figsize=(6, 6.5))
            formaciones_vistas = set()
            for _, capa_geo in perfil.iterrows():
                formacion = capa_geo["Formacion"] if pd.notna(capa_geo["Formacion"]) else "Sin dato"
                color = color_formacion.get(formacion, "#999999")
                etiqueta = formacion if formacion not in formaciones_vistas else None
                ax.bar(
                    0, capa_geo["Fin"] - capa_geo["Inicio"], bottom=capa_geo["Inicio"],
                    width=0.5, color=color, edgecolor="black", linewidth=0.6, label=etiqueta,
                )
                formaciones_vistas.add(formacion)

            ax.invert_yaxis()
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel("Profundidad (m)")
            ax.set_title(f"Columna litológica — {pozo_lito_sel}", fontweight="bold", pad=14)
            ax.legend(title="Formación", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
            sns.despine(ax=ax, bottom=True)
            ax.spines["left"].set_color("black")
            ax.minorticks_on()
            ax.tick_params(axis="y", which="both", color="black")
            fig.subplots_adjust(right=0.62, top=0.90)
            st.pyplot(fig)

        with st.expander("Ver detalle de capas de este pozo"):
            st.dataframe(
                _para_mostrar(
                    perfil[["Inicio", "Fin", "Litologia", "Formacion", "Fuente BD", "Observaciones"]]
                ),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Tab 3 — Hidroquímica interactiva: mapa por parámetro medido
# ---------------------------------------------------------------------------
with tab_hidroquimica:
    st.header("🧪 Hidroquímica interactiva — Mapa por parámetro medido")
    st.markdown("Elegí un parámetro hidroquímico para ver, en el mapa, qué pozos tienen ese dato medido.")

    hidroquimica_cruda = datos["hidroquimica"].copy()
    columnas_metadata_hq = {"Cod UK", "Fecha", "Año", "Observaciones", "Fuente", "Fuente.1"}
    parametros_disponibles = [c for c in hidroquimica_cruda.columns if c not in columnas_metadata_hq]

    parametro_sel = st.selectbox(
        "Parámetro hidroquímico", options=sorted(parametros_disponibles), key="parametro_hq_sel"
    )

    con_parametro = hidroquimica_cruda.dropna(subset=[parametro_sel])
    cod_uk_con_parametro = set(con_parametro["Cod UK"].unique())

    pozos_hq = pozos_f[pozos_f["lat"].notna() & pozos_f["lon"].notna()].copy()
    pozos_hq["tiene_parametro"] = pozos_hq["Cod UK"].isin(cod_uk_con_parametro)

    if pozos_hq.empty:
        st.info("No hay pozos con coordenadas para el filtro seleccionado.")
    else:
        con_dato = int(pozos_hq["tiene_parametro"].sum())
        st.caption(f"🟣 {con_dato:,} de {len(pozos_hq):,} pozos (filtro actual) tienen '{parametro_sel}' medido.")

        mapa_hq = pozos_hq.rename(
            columns={"Cod UK": "cod_uk", "acuifero": "acuifero_pozo", "Fuente BD": "fuente_bd"}
        )
        mapa_hq["color"] = mapa_hq["tiene_parametro"].apply(
            lambda tiene: [153, 50, 204, 220] if tiene else [180, 180, 180, 90]
        )
        mapa_hq["radio"] = mapa_hq["tiene_parametro"].apply(lambda tiene: 1000 if tiene else 500)

        capa_hq = pdk.Layer(
            "ScatterplotLayer", data=mapa_hq,
            get_position=["lon", "lat"], get_fill_color="color",
            get_line_color=[0, 0, 0, 180], line_width_min_pixels=0.6,
            stroked=True, get_radius="radio", pickable=True, auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[capa_hq],
            initial_view_state=pdk.ViewState(
                latitude=float(mapa_hq["lat"].mean()), longitude=float(mapa_hq["lon"].mean()), zoom=6.5,
            ),
            map_style=MAPA_ESTILO,
            tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
        ))
        st.caption("🟣 Pozo con el parámetro medido. ⚪ Pozo sin ese dato (filtro actual).")

        if con_dato > 0:
            valores = pd.to_numeric(con_parametro[parametro_sel], errors="coerce").dropna()
            if not valores.empty:
                fig, ax = plt.subplots(figsize=(7, 3.2))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                sns.histplot(valores, bins=20, color="#984ea3", edgecolor="black", ax=ax)

                if parametro_sel == "pH":
                    for limite in UMBRAL_PH:
                        ax.axvline(
                            limite, color="#EF4444", linestyle="--", linewidth=1.5,
                            label="Rango normativo (6.5–8.5)" if limite == UMBRAL_PH[0] else None,
                        )
                    ax.legend(fontsize=8)
                elif parametro_sel in UMBRALES_NORMATIVOS:
                    umbral = UMBRALES_NORMATIVOS[parametro_sel]
                    ax.axvline(
                        umbral, color="#EF4444", linestyle="--", linewidth=1.5,
                        label=f"Límite normativo ({umbral:g})",
                    )
                    ax.legend(fontsize=8)

                ax.set_title(f"Distribución de {parametro_sel} (todos los pozos con dato)", fontweight="bold")
                ax.set_xlabel(_limpiar_texto(parametro_sel))
                ax.set_ylabel("Cantidad de mediciones")
                _estilizar_ejes(ax)
                ax.grid(False)
                plt.tight_layout()
                st.pyplot(fig, transparent=True)

                if parametro_sel in UMBRALES_NORMATIVOS or parametro_sel == "pH":
                    st.caption(
                        "Línea roja punteada: límite/rango de referencia para agua de consumo "
                        "(Decreto 253/79 y normativa UNIT vigente en Uruguay)."
                    )

    with st.expander("Ver tabla cruda de Hidroquímica"):
        hidroquimica_mostrar = hidroquimica_cruda.copy()
        columnas_numericas_hq = hidroquimica_mostrar.select_dtypes(include="number").columns
        hidroquimica_mostrar[columnas_numericas_hq] = hidroquimica_mostrar[columnas_numericas_hq].round(2)
        st.dataframe(_para_mostrar(hidroquimica_mostrar), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 — Hidráulica: profundidad, niveles y caudal
# ---------------------------------------------------------------------------
with tab_hidraulica:
    st.header("💧 Hidráulica — Profundidad, niveles y caudal")
    st.markdown(
        "Mapa y ficha de pozos con datos hidráulicos: profundidad total, nivel "
        "estático (NE), nivel dinámico (ND) y caudal de explotación. "
        "*Nota: \"Volumen anual\" y \"Criterio\" no están cargados en la base actual — "
        "si me pasás esos datos los sumamos.*"
    )

    cod_uk_con_hidraulica = set(hidraulica_ultima["Cod UK"].unique())
    pozos_hidraulica = pozos_f[
        pozos_f["Cod UK"].isin(cod_uk_con_hidraulica) & pozos_f["lat"].notna() & pozos_f["lon"].notna()
    ]

    if pozos_hidraulica.empty:
        st.info("No hay pozos con datos hidráulicos para el filtro seleccionado.")
    else:
        opciones_hid = sorted(pozos_hidraulica["Cod UK"].unique().tolist())
        pozo_hid_sel = st.selectbox(
            "Elegí un pozo con datos hidráulicos", options=opciones_hid, key="pozo_hid_sel"
        )

        mapa_hid = pozos_hidraulica.rename(
            columns={"Cod UK": "cod_uk", "acuifero": "acuifero_pozo", "Fuente BD": "fuente_bd"}
        ).copy()
        mapa_hid["es_seleccionado"] = mapa_hid["cod_uk"] == pozo_hid_sel
        mapa_hid["color"] = mapa_hid["es_seleccionado"].apply(
            lambda sel: [230, 25, 25, 255] if sel else [30, 100, 180, 140]
        )
        mapa_hid["radio"] = mapa_hid["es_seleccionado"].apply(lambda sel: 1800 if sel else 700)
        fila_hid_sel = mapa_hid[mapa_hid["es_seleccionado"]].iloc[0]

        col_mapa_hid, col_ficha = st.columns([1, 1.2])

        with col_mapa_hid:
            st.markdown("**Pozos con datos hidráulicos**")
            capa_hid = pdk.Layer(
                "ScatterplotLayer", data=mapa_hid,
                get_position=["lon", "lat"], get_fill_color="color",
                get_line_color=[0, 0, 0, 180], line_width_min_pixels=0.6,
                stroked=True, get_radius="radio", pickable=True, auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[capa_hid],
                initial_view_state=pdk.ViewState(
                    latitude=float(fila_hid_sel["lat"]), longitude=float(fila_hid_sel["lon"]), zoom=9,
                ),
                map_style=MAPA_ESTILO,
                tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
            ))
            st.caption("🔴 Pozo seleccionado. 🔵 Resto de los pozos con datos hidráulicos (filtro actual).")

        with col_ficha:
            fila_h = hidraulica_ultima[hidraulica_ultima["Cod UK"] == pozo_hid_sel].iloc[0]
            profundidad_total = fila_hid_sel.get("Prof. Total")

            st.markdown(f"**Ficha hidráulica — {pozo_hid_sel}**")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Profundidad total",
                f"{profundidad_total:.1f} m" if pd.notna(profundidad_total) else "s/d",
            )
            m2.metric("Nivel estático (NE)", f"{fila_h['NE_m']:.1f} m" if pd.notna(fila_h["NE_m"]) else "s/d")
            m3.metric("Nivel dinámico (ND)", f"{fila_h['ND-m']:.1f} m" if pd.notna(fila_h["ND-m"]) else "s/d")
            st.metric("Caudal", f"{fila_h['Caudal']:.1f} m³/h" if pd.notna(fila_h["Caudal"]) else "s/d")

            if pd.notna(profundidad_total):
                fig, ax = plt.subplots(figsize=(4, 6))
                ax.bar(0, profundidad_total, width=0.4, color="#d9d9d9", edgecolor="black")
                if pd.notna(fila_h["NE_m"]):
                    ax.axhline(fila_h["NE_m"], color="#377eb8", linewidth=2.2, label="NE (nivel estático)")
                if pd.notna(fila_h["ND-m"]):
                    ax.axhline(fila_h["ND-m"], color="#e41a1c", linewidth=2.2, label="ND (nivel dinámico)")
                ax.invert_yaxis()
                ax.set_xlim(-0.5, 0.5)
                ax.set_xticks([])
                ax.set_ylabel("Profundidad (m)")
                ax.set_title(f"Esquema — {pozo_hid_sel}", fontweight="bold", pad=14)
                ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
                sns.despine(ax=ax, bottom=True)
                ax.spines["left"].set_color("black")
                fig.subplots_adjust(right=0.55, top=0.90)
                st.pyplot(fig)
            else:
                st.caption("Este pozo no tiene profundidad total cargada para dibujar el esquema.")

        with st.expander("Ver historial hidráulico completo de este pozo"):
            historial = hidraulica_cruda[hidraulica_cruda["Cod UK"] == pozo_hid_sel].sort_values("Año")
            st.dataframe(_para_mostrar(historial), use_container_width=True)

    with st.expander("Ver tabla cruda de Hidráulica (todos los pozos)"):
        st.dataframe(_para_mostrar(hidraulica_cruda), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5 — Declaraciones Juradas
# ---------------------------------------------------------------------------
with tab_dj:
    st.header("📝 Pozos con y sin Declaración Jurada")
    st.markdown(
        "Se considera que un pozo **tiene declaración jurada** cuando su \"Codigo Fuente\" "
        "incluye un código SGRH (Sistema de Gestión de Recursos Hídricos), que DINAGUA asigna "
        "al registrar una declaración jurada. Si tenés otro criterio o columna para esto, avisame."
    )

    pozos_dj = pozos_f[pozos_f["lat"].notna() & pozos_f["lon"].notna()].copy()
    pozos_dj["tiene_dj"] = pozos_dj["codigo_sgrh"].notna()

    con_dj = int(pozos_dj["tiene_dj"].sum())
    sin_dj = len(pozos_dj) - con_dj
    m1, m2 = st.columns(2)
    m1.metric("Con declaración jurada (SGRH)", f"{con_dj:,}")
    m2.metric("Sin declaración jurada", f"{sin_dj:,}")

    if pozos_dj.empty:
        st.info("No hay pozos con coordenadas para el filtro seleccionado.")
    else:
        mapa_dj = pozos_dj.rename(
            columns={"Cod UK": "cod_uk", "acuifero": "acuifero_pozo", "Fuente BD": "fuente_bd"}
        ).copy()
        mapa_dj["color"] = mapa_dj["tiene_dj"].apply(
            lambda tiene: [50, 160, 50, 220] if tiene else [220, 120, 20, 180]
        )
        capa_dj = pdk.Layer(
            "ScatterplotLayer", data=mapa_dj,
            get_position=["lon", "lat"], get_fill_color="color",
            get_line_color=[0, 0, 0, 180], line_width_min_pixels=0.6,
            stroked=True, get_radius=800, pickable=True, auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[capa_dj],
            initial_view_state=pdk.ViewState(
                latitude=float(mapa_dj["lat"].mean()), longitude=float(mapa_dj["lon"].mean()), zoom=6.5,
            ),
            map_style=MAPA_ESTILO,
            tooltip={"text": "Pozo: {cod_uk}\nAcuífero: {acuifero_pozo}\nFuente BD: {fuente_bd}"},
        ))
        st.caption("🟢 Con declaración jurada (código SGRH). 🟠 Sin declaración jurada.")

        resumen_dj = (
            pozos_dj.groupby(["acuifero", "tiene_dj"], dropna=False)
            .size()
            .reset_index(name="cantidad_pozos")
        )
        resumen_dj["Declaración jurada"] = resumen_dj["tiene_dj"].map({True: "Con DJ", False: "Sin DJ"})
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=resumen_dj, x="acuifero", y="cantidad_pozos", hue="Declaración jurada",
            palette={"Con DJ": "#4daf4a", "Sin DJ": "#ff7f00"}, ax=ax,
        )
        ax.set_title("Pozos con/sin declaración jurada, por acuífero", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Cantidad de pozos")
        ax.legend(title="", fontsize=9)
        _estilizar_ejes(ax)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Ver tabla completa"):
        columnas_dj = ["Cod UK", "acuifero", "Fuente BD", "Codigo Fuente", "codigo_sgrh", "tiene_dj"]
        st.dataframe(_para_mostrar(pozos_dj[columnas_dj]), use_container_width=True)
