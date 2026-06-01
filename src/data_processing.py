"""Carga, limpieza y exportación del dataset de estudiantes."""

from pathlib import Path

import numpy as np
import pandas as pd


def cargar_y_limpiar_datos(path_csv: str | Path) -> pd.DataFrame:
    """
    Carga el CSV de estudiantes, imputa nulos, recorta outliers en ausencias
    y conserva nivel_socioeconomico como variable numérica (1-4).
    """
    path_csv = Path(path_csv)
    df = pd.read_csv(path_csv)

    columnas_categoricas = ["nivel_socioeconomico"]
    columnas_numericas = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in columnas_categoricas
    ]

    for col in columnas_numericas:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in columnas_categoricas:
        if col in df.columns and df[col].isna().any():
            moda = df[col].mode(dropna=True)
            valor_moda = moda.iloc[0] if not moda.empty else df[col].iloc[0]
            df[col] = df[col].fillna(valor_moda)

    df["nivel_socioeconomico"] = df["nivel_socioeconomico"].astype(int)

    percentil_99 = df["ausencias"].quantile(0.99)
    df["ausencias"] = df["ausencias"].clip(upper=percentil_99)

    path_salida = path_csv.parent / "student_data_limpio.csv"
    df.to_csv(path_salida, index=False)

    return df


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    ruta_entrada = raiz / "data" / "student_data.csv"
    df_limpio = cargar_y_limpiar_datos(ruta_entrada)
    print(f"Filas: {len(df_limpio)}, columnas: {len(df_limpio.columns)}")
    print(f"Guardado en: {raiz / 'data' / 'student_data_limpio.csv'}")
