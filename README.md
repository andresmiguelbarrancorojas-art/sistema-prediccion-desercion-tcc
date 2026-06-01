# Sistema Predictivo de Retención Estudiantil

**Trabajo de Conclusión de Curso (TCC) — Ingeniería de Software**

Herramienta de apoyo a la toma de decisiones académicas que estima la probabilidad de deserción estudiantil a partir de indicadores operativos disponibles en la institución. El sistema permite evaluar casos individuales y procesar lotes de estudiantes mediante una interfaz web desarrollada con Streamlit.

---

## Descripción y enfoque preventivo

La deserción estudiantil es un problema multifactorial que impacta la continuidad formativa y los recursos institucionales. En lugar de actuar únicamente cuando el estudiante ya ha abandonado el programa, este proyecto adopta un **enfoque preventivo**: identificar de forma temprana perfiles con señales de riesgo para activar protocolos de acompañamiento, tutoría y seguimiento académico.

El flujo de la solución comprende tres etapas:

1. **Preparación de datos** — limpieza, imputación de valores faltantes y tratamiento de valores atípicos.
2. **Modelado predictivo** — entrenamiento supervisado con balanceo de clases (SMOTE) y ajuste de umbral orientado a maximizar la detección de estudiantes en riesgo.
3. **Despliegue operativo** — aplicación web para consultas individuales y cargas masivas en formato CSV.

---

## Vector de características

El modelo utiliza un vector de tres variables numéricas, alineadas con el orden de entrenamiento:

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `ausencias` | Entero | Número de inasistencias registradas en el período de observación. |
| `nota_periodo_anterior` | Float | Calificación obtenida en el período académico previo (escala 0.0 – 5.0). |
| `nivel_socioeconomico` | Entero | Estrato socioeconómico codificado en escala ordinal 1 – 4. |

**Variable objetivo:** `desercion` (0 = estable, 1 = riesgo de deserción).

---

## Algoritmo de machine learning

Se emplea **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) como estimador principal, configurado con:

- `n_estimators=300`
- `class_weight="balanced"` para mitigar el desbalance entre clases
- `random_state=42` para reproducibilidad

Durante el entrenamiento se comparan también modelos de Regresión Logística; el algoritmo con mejor **F1-Score en la clase de riesgo** se selecciona para exportación. El pipeline incluye:

- **SMOTE** (`imbalanced-learn`) para balancear el conjunto de entrenamiento.
- **Ajuste de umbral de decisión** sobre el conjunto de validación, priorizando la detección de casos en riesgo (clase 1).

El artefacto resultante se persiste en `modelo_entrenado.pkl` en la raíz del proyecto.

---

## Estructura del repositorio

```
sistema-retencion/
├── data/
│   ├── student_data.csv          # Dataset original
│   └── student_data_limpio.csv   # Dataset tras limpieza
├── src/
│   ├── data_processing.py        # Carga, limpieza y exportación
│   └── train_model.py            # Entrenamiento y evaluación
├── app.py                        # Interfaz Streamlit
├── modelo_entrenado.pkl          # Modelo serializado (generado al entrenar)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requisitos del entorno

- **Python** 3.9 o superior
- **pip** actualizado
- Conexión a internet (solo para la instalación de dependencias)

---

## Instalación

Clona el repositorio y crea un entorno virtual aislado:

```bash
git clone https://github.com/<usuario>/sistema-retencion.git
cd sistema-retencion

python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Ejecución del pipeline

### 1. Limpieza de datos

```bash
python src/data_processing.py
```

Genera `data/student_data_limpio.csv` a partir de `data/student_data.csv`.

### 2. Entrenamiento del modelo

```bash
python src/train_model.py
```

Entrena, evalúa y exporta `modelo_entrenado.pkl` en la raíz del proyecto.

### 3. Interfaz web (Streamlit)

```bash
python -m streamlit run app.py
```

Abre el navegador en **http://localhost:8501**. La aplicación ofrece dos modos:

- **Predicción individual** — formulario con sliders para evaluar un estudiante.
- **Predicción masiva** — carga de CSV con las columnas `ausencias`, `nota_periodo_anterior` y `nivel_socioeconomico`.

> **Nota:** Si `modelo_entrenado.pkl` no existe, la aplicación mostrará un error indicando que debe ejecutarse primero el entrenamiento.

---

## Despliegue

### Local (desarrollo o demostración)

Sigue los pasos de instalación y ejecución anteriores. Para entornos sin interfaz gráfica:

```bash
python -m streamlit run app.py --server.headless true
```

### Streamlit Community Cloud

1. Sube el repositorio a GitHub (incluyendo `modelo_entrenado.pkl` y `requirements.txt`).
2. Inicia sesión en [share.streamlit.io](https://share.streamlit.io).
3. Conecta el repositorio y define `app.py` como archivo principal.
4. Confirma que `requirements.txt` esté en la raíz; Streamlit instalará las dependencias automáticamente.

### Consideraciones de producción

- No versionar secretos ni archivos `.streamlit/secrets.toml` (ya excluidos en `.gitignore`).
- Regenerar el modelo cuando se actualice el dataset de entrenamiento.
- Validar periódicamente las métricas de recall y F1 sobre datos recientes.

---

## Dependencias principales

| Paquete | Uso |
|---------|-----|
| `pandas` | Manipulación de datos tabulares |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Random Forest, métricas y partición de datos |
| `imbalanced-learn` | Balanceo SMOTE |
| `streamlit` | Interfaz web interactiva |

---

## Licencia

Proyecto académico desarrollado con fines educativos en el marco del TCC de Ingeniería de Software.
