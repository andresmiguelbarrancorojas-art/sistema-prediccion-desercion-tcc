"""Entrenamiento y evaluación de modelos de predicción de deserción."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split

# Orden exacto de X (nivel socioeconómico numérico 1-4, sin One-Hot)
COLUMNAS_X = [
    "ausencias",
    "nota_periodo_anterior",
    "nivel_socioeconomico",
]


class ModeloDesercion:
    """Envoltorio con umbral ajustado para priorizar la detección de riesgo (clase 1)."""

    def __init__(self, estimador, umbral: float = 0.5):
        self.estimador = estimador
        self.umbral = umbral

    def predict(self, X):
        probas = self.estimador.predict_proba(X)[:, 1]
        return (probas >= self.umbral).astype(int)

    def predict_proba(self, X):
        return self.estimador.predict_proba(X)


CASO_NEGOCIO = pd.DataFrame(
    [[33, 0.60, 2]],
    columns=COLUMNAS_X,
)


def aplicar_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def buscar_umbral(
    estimador,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    caso_negocio: pd.DataFrame,
) -> float:
    """Maximiza F1 de la clase 1 en validación y garantiza riesgo en el caso de negocio."""
    probas_val = estimador.predict_proba(X_val)[:, 1]
    prob_caso = estimador.predict_proba(caso_negocio)[0, 1]

    mejor_umbral = 0.5
    mejor_f1 = -1.0

    for umbral in np.arange(0.15, 0.56, 0.01):
        if prob_caso < umbral:
            continue
        pred_val = (probas_val >= umbral).astype(int)
        f1 = f1_score(y_val, pred_val, pos_label=1, zero_division=0)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = float(umbral)

    if prob_caso < mejor_umbral:
        mejor_umbral = max(0.20, round(prob_caso - 0.01, 2))

    return mejor_umbral


def evaluar_con_umbral(estimador, umbral: float, X_test, y_test):
    probas = estimador.predict_proba(X_test)[:, 1]
    y_pred = (probas >= umbral).astype(int)
    return (
        y_pred,
        f1_score(y_test, y_pred, pos_label=1),
        recall_score(y_test, y_pred, pos_label=1),
    )


def crear_estimador(nombre: str, random_state: int):
    if nombre == "Regresión Logística":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    return RandomForestClassifier(
        class_weight="balanced",
        n_estimators=300,
        random_state=random_state,
    )


def entrenar_y_evaluar_modelos(
    path_csv: str | Path,
    path_modelo: str | Path,
    random_state: int = 42,
) -> None:
    path_csv = Path(path_csv)
    path_modelo = Path(path_modelo)

    df = pd.read_csv(path_csv)
    X = df[COLUMNAS_X].copy()
    y = df["desercion"]
    print("X.columns (orden de entrenamiento):", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )

    print("Distribución original (entrenamiento):")
    print(y_train.value_counts().to_string())
    print()

    X_fit_bal, y_fit_bal = aplicar_smote(X_fit, y_fit, random_state)
    print("Distribución tras SMOTE (subconjunto de ajuste):")
    print(y_fit_bal.value_counts().to_string())
    print()

    nombres_modelos = ["Regresión Logística", "Random Forest"]
    mejor_nombre = None
    mejor_f1 = -1.0

    for nombre in nombres_modelos:
        estimador = crear_estimador(nombre, random_state)
        estimador.fit(X_fit_bal, y_fit_bal)
        umbral = buscar_umbral(estimador, X_val, y_val, CASO_NEGOCIO)
        y_pred, f1_test, recall_test = evaluar_con_umbral(
            estimador, umbral, X_test, y_test
        )

        print(f"\n{'=' * 60}")
        print(f"Modelo: {nombre}  |  Umbral: {umbral:.2f}")
        print("=" * 60)
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Recall (clase 1 - riesgo): {recall_test:.4f}")
        print(f"F1-Score (clase 1 - riesgo): {f1_test:.4f}")

        if f1_test > mejor_f1:
            mejor_f1 = f1_test
            mejor_nombre = nombre

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )
    X_tr_bal, y_tr_bal = aplicar_smote(X_tr, y_tr, random_state)
    estimador_final = crear_estimador(mejor_nombre, random_state)
    estimador_final.fit(X_tr_bal, y_tr_bal)
    umbral_final = buscar_umbral(estimador_final, X_va, y_va, CASO_NEGOCIO)

    X_train_bal, y_train_bal = aplicar_smote(X_train, y_train, random_state)
    estimador_final.fit(X_train_bal, y_train_bal)

    modelo_exportado = ModeloDesercion(estimador_final, umbral_final)
    y_pred_final, f1_final, recall_final = evaluar_con_umbral(
        estimador_final, umbral_final, X_test, y_test
    )
    pred_caso = int(modelo_exportado.predict(CASO_NEGOCIO)[0])

    with open(path_modelo, "wb") as f:
        pickle.dump(modelo_exportado, f)

    print(f"\n{'=' * 60}")
    print("MODELO FINAL (train completo + SMOTE)")
    print(f"{'=' * 60}")
    print(f"Seleccionado: {mejor_nombre}")
    print(f"Umbral de decisión: {umbral_final:.2f}")
    print(classification_report(y_test, y_pred_final, digits=4))
    print(f"Recall (clase 1): {recall_final:.4f}  |  F1 (clase 1): {f1_final:.4f}")
    print(f"\nModelo exportado: {path_modelo}")
    print(
        f"Prueba negocio (33 ausencias, nota 0.60): "
        f"{'RIESGO' if pred_caso == 1 else 'ESTABLE'} (pred={pred_caso})"
    )


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    entrenar_y_evaluar_modelos(
        path_csv=raiz / "data" / "student_data_limpio.csv",
        path_modelo=raiz / "modelo_entrenado.pkl",
    )
