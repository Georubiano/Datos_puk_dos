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

pozos_f = pozos[pozos["acuifero"].isin(acuifero_sel) & pozos["Fuente BD"].isin(fuentes_sel)]
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

# Paleta ColorBrewer "Set1" (cualitativa), una entrada por Fuente BD. Se usa
# tanto en el mapa como en el gráfico de cobertura, para que el color de
# cada fuente sea siempre el mismo en toda la app.
SET1 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]
FUENTES_ORDEN = ["Dinagua", "Dinamige", "OSE", "Privado", "MEVIR", "Academia", "Ancap", "Sin dato"]
COLOR_FUENTE = {fuente: SET1[i % len(SET1)] for i, fuente in enumerate(FUENTES_ORDEN)}


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
        df[col] = df[col].astype(str).replace("nan", "")
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
        lambda f: _hex_a_rgb(COLOR_FUENTE.get(f, "#999999")) + [200]
    )
    mapa_df = con_coords.rename(columns={
        "Cod UK": "cod_uk", "Fuente BD": "fuente_bd", "acuifero": "acuifero_pozo",
    })[["lat", "lon", "color", "cod_uk", "acuifero_pozo", "fuente_bd", "tiene_litologia", "tiene_hidroquimica"]]

    fuentes_presentes = sorted(con_coords["Fuente BD"].dropna().unique().tolist())
    st.markdown("**Leyenda — color por Fuente BD:**")
    leyenda_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;'>"
    for fuente in fuentes_presentes:
        hex_c = COLOR_FUENTE.get(fuente, "#999999")
        leyenda_html += (
            f"<span style='background:{hex_c};color:white;border:1px solid black;"
            f"padding:3px 10px;border-radius:12px;font-size:12px;'>{fuente}</span>"
        )
    leyenda_html += "</div>"
    st.markdown(leyenda_html, unsafe_allow_html=True)
    st.caption(f"📍 {len(mapa_df):,} pozos georreferenciados (puntos con borde negro, color = Fuente BD).")

    capa = pdk.Layer(
        "ScatterplotLayer", data=mapa_df,
        get_position=["lon", "lat"], get_fill_color="color",
        get_line_color=[0, 0, 0], line_width_min_pixels=1,
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
            get_line_color=[0, 0, 0], line_width_min_pixels=1,
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
