# Pozos - Sistema Acuífero Raigón y Sistema Acuífero Guaraní (DINAGUA)

Panel de indicadores y mapa interactivo a partir de la base consolidada
`base_final.xlsx` (1851 pozos, 9 hojas relacionadas por `Cod UK`).

## Estructura

```
raigon_guarani/
├── data/raw/base_final.xlsx
├── src/
│   ├── data_loading.py   # lectura de las 9 hojas del Excel
│   ├── cleaning.py       # normalización + clasificación por acuífero
│   ├── geo_utils.py      # UTM 21S -> lat/lon (sin geopandas, más liviano)
│   └── eda.py            # indicadores (KPIs) para la app
├── app/streamlit_app.py  # app principal
└── requirements.txt
```

## Puntos de criterio de domino a revisar con Georgina

- **Clasificación de acuífero**: se descartó usar `Departamento` como
  criterio principal porque tiene errores de carga — 35 pozos cuyo
  Departamento no coincide con su propia ubicación (ej. 14 pozos
  etiquetados "San José" que están geográficamente en la zona de
  Tacuarembó, y 13 etiquetados "Tacuarembó" que están en la zona de
  Montevideo/San José). En su lugar, `acuifero` se calcula por
  **latitud real** del pozo (`cleaning.clasificar_acuifero_por_ubicacion`):
  hay un vacío claro de pozos entre -34.3° y -32.0° de latitud, así que
  -33.0° separa el cluster sur (Raigón) del cluster norte (Guaraní) sin
  ambigüedad. El Departamento crudo queda como `acuifero_departamento`
  (diagnóstico) y como respaldo solo para los pozos sin coordenadas
  válidas. Cruce adicional posible: la `Formación` geológica relevada en
  Litología (más ruidosa, porque un mismo pozo puede tener varias
  formaciones a distintas profundidades).
- **Coordenadas fuera de rango**: 1 pozo (`UK000681`) tiene una UTM con
  un error de tipeo que lo ubica en medio del Atlántico. `geo_utils
  .marcar_coordenadas_fuera_de_rango` detecta esto (fuera del bounding
  box de Uruguay) y le pone `lat`/`lon` en blanco — no aparece en el
  mapa, pero sigue en las tablas/indicadores. Vale la pena revisar esa
  fila en el Excel original.
- **Vínculo con el proyecto "Libertad"**: 375 de 1851 pozos tienen un
  código `SGRH-...` embebido en `Codigo Fuente`, el mismo formato usado
  en el catastro Raigón de Libertad. Queda extraído en la columna
  `codigo_sgrh`, listo para cruzar cuando se quiera.
- **"Tiene dato" en Litología/Hidroquímica/Hidráulica/Histórico**
  significa "aparece al menos una vez para ese pozo en esa hoja", no
  necesariamente que el dato esté completo o sea de buena calidad.

## Correr localmente

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
