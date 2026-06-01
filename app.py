"""Interfaz Streamlit — Sistema Predictivo de Retención Estudiantil."""

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "modelo_entrenado.pkl"

COLUMNAS_ENTRENAMIENTO = [
    "ausencias",
    "nota_periodo_anterior",
    "nivel_socioeconomico",
]

sys.path.insert(0, str(RAIZ / "src"))
from train_model import ModeloDesercion  # noqa: E402


@st.cache_resource
def cargar_modelo():
    with open(RUTA_MODELO, "rb") as f:
        return pickle.load(f)


def etiqueta_riesgo(prediccion: int) -> str:
    return "Estudiante en Riesgo de Deserción" if prediccion == 1 else "Estudiante Estable"


st.set_page_config(
    page_title="Retención Estudiantil",
    page_icon="🎓",
    layout="wide",
)

st.title("Sistema Predictivo de Retención Estudiantil")
st.markdown(
    "Herramienta de apoyo para identificar estudiantes con probabilidad de deserción "
    "a partir de ausencias, rendimiento académico y nivel socioeconómico (escala 1-4)."
)

if not RUTA_MODELO.exists():
    st.error(
        "No se encontró `modelo_entrenado.pkl`. Ejecuta primero: `python src/train_model.py`"
    )
    st.stop()

modelo = cargar_modelo()
model = modelo.estimador if isinstance(modelo, ModeloDesercion) else modelo

tab_individual, tab_masiva = st.tabs(
    ["Predicción Individual", "Predicción Masiva (CSV)"]
)

with tab_individual:
    st.subheader("Evaluación de un estudiante")
    st.caption("Ingresa los datos del alumno y obtén una predicción inmediata.")

    col1, col2, col3 = st.columns(3)

    with col1:
        ausencias = st.slider("Ausencias", min_value=0, max_value=40, value=10)
    with col2:
        nota = st.slider(
            "Nota período anterior",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
        )
    with col3:
        nivel = st.selectbox(
            "Nivel socioeconómico (1-4)",
            options=[1, 2, 3, 4],
            index=1,
        )

    if st.button("Predecir Riesgo", type="primary", use_container_width=True):
        ausencias = int(ausencias)
        nota_periodo_anterior = float(nota)
        nivel = int(nivel)

        df_input = pd.DataFrame(
            [[ausencias, nota_periodo_anterior, nivel]],
            columns=COLUMNAS_ENTRENAMIENTO,
        )

        prob_riesgo = float(model.predict_proba(df_input)[0][1])
        prediccion = int(model.predict(df_input)[0])

        st.divider()
        st.metric(
            "Probabilidad de deserción (clase 1)",
            f"{prob_riesgo * 100:.2f}%",
        )

        if prediccion == 1:
            st.error(f"⚠️ **{etiqueta_riesgo(prediccion)}**")
            st.markdown(
                "Se recomienda activar protocolos de acompañamiento y seguimiento académico."
            )
        else:
            st.success(f"✅ **{etiqueta_riesgo(prediccion)}**")
            st.markdown("El perfil actual no presenta señales críticas de deserción.")

        with st.expander("Detalle de la predicción"):
            st.dataframe(df_input, use_container_width=True, hide_index=True)

with tab_masiva:
    st.subheader("Carga masiva desde CSV")
    st.caption(
        "El archivo debe incluir: `ausencias`, `nota_periodo_anterior`, "
        "`nivel_socioeconomico` (1-4). Opcional: `id_estudiante`."
    )

    archivo = st.file_uploader(
        "Arrastra o selecciona un archivo CSV",
        type=["csv"],
    )

    if archivo is not None:
        df_entrada = pd.read_csv(archivo)
        requeridas = {"ausencias", "nota_periodo_anterior", "nivel_socioeconomico"}
        faltantes = requeridas - set(df_entrada.columns)

        if faltantes:
            st.warning(
                f"Faltan columnas obligatorias: {', '.join(sorted(faltantes))}"
            )
        else:
            df_input = pd.DataFrame(
                df_entrada[list(COLUMNAS_ENTRENAMIENTO)].values,
                columns=COLUMNAS_ENTRENAMIENTO,
            )
            predicciones = model.predict(df_input)
            probas_riesgo = model.predict_proba(df_input)[:, 1]

            df_resultado = df_entrada.copy()
            df_resultado["probabilidad_desercion"] = probas_riesgo
            df_resultado["prediccion"] = predicciones.astype(int)
            df_resultado["riesgo"] = df_resultado["prediccion"].map(
                lambda x: "Riesgo de deserción" if x == 1 else "Estable"
            )

            total_riesgo = int((df_resultado["prediccion"] == 1).sum())
            total_estable = len(df_resultado) - total_riesgo

            m1, m2, m3 = st.columns(3)
            m1.metric("Registros procesados", len(df_resultado))
            m2.metric("En riesgo", total_riesgo)
            m3.metric("Estables", total_estable)

            st.dataframe(df_resultado, use_container_width=True, hide_index=True)
