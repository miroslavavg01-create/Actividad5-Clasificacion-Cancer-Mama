"""Funciones de preparación para Breast Cancer Wisconsin Diagnostic."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

FEATURE_NAMES = list(load_breast_cancer().feature_names)
TARGET_MAP = {"benigno": 0, "maligno": 1}

def obtener_datos_publicos():
    """Carga la copia del dataset público incluida en scikit-learn."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.data.copy()
    df.insert(0, "diagnostico", bunch.target.map({0: "maligno", 1: "benigno"}).values)
    return df

def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nombres, elimina duplicados y codifica la variable objetivo.

    No se escala aquí para evitar fuga de información; StandardScaler se ajusta
    dentro del Pipeline usando únicamente los folds de entrenamiento.
    """
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)
    out["diagnostico"] = out["diagnostico"].astype(str).str.strip().str.lower()
    out["target_maligno"] = out["diagnostico"].map(TARGET_MAP)
    if out["target_maligno"].isna().any():
        raise ValueError("Valores de diagnóstico no reconocidos")
    return out

def generar_archivos(base="."):
    base = Path(base)
    ini = base / "datos" / "datos_ini"
    limp = base / "datos" / "datos_limp"
    ini.mkdir(parents=True, exist_ok=True); limp.mkdir(parents=True, exist_ok=True)
    original = obtener_datos_publicos()
    original.to_csv(ini / "wdbc_original.csv", index=False)
    limpio = preparar_dataframe(original)
    limpio.to_csv(limp / "wdbc_limpio.csv", index=False)
    return original, limpio
