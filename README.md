# Actividad 5 — Preparación, entrenamiento y MLflow

## Objetivo
Comparar regresión logística y Random Forest en el dataset público Breast Cancer Wisconsin (Diagnostic). Este ejercicio es académico y no es una herramienta de diagnóstico médico.

## Dataset y fuente
El conjunto incluye 569 observaciones, 30 características numéricas y un diagnóstico benigno/maligno. Se utiliza la copia distribuida por scikit-learn del dataset de UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic. Las variables describen características calculadas de núcleos celulares a partir de imágenes digitalizadas de aspiraciones con aguja fina.

## Estructura
- `datos/datos_ini/`: copia inicial del CSV público.
- `datos/datos_limp/`: CSV limpio con variable objetivo codificada.
- `fuentes/entrena.ipynb`: notebook auto-contenido para Google Colab.
- `fuentes/datos_prep.py`: funciones de limpieza.
- `fuentes/train.py`: entrenamiento, GridSearchCV y seguimiento MLflow.
- `resultados/`: métricas y gráficas de una ejecución de referencia local; al repetir el notebook se generan los resultados de la ejecución de Colab.
- `mlruns/`: seguimiento local de MLflow creado cuando se ejecuta `train.py` con MLflow instalado.

## Entorno
Python 3.10 o posterior recomendado. Instala dependencias con `pip install -r requirements.txt`. En Colab ejecuta `fuentes/entrena.ipynb` de arriba hacia abajo; el notebook instala dependencias en una celda. GPU no es necesaria para estos modelos tabulares; la ejecución se hace en CPU sin problemas.

## Ejecución local
Desde la carpeta raíz del repositorio:
```bash
pip install -r requirements.txt
python fuentes/train.py
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```
Después abre `http://127.0.0.1:5000` para comparar los experimentos registrados. Para revisar el cuaderno: abre `fuentes/entrena.ipynb` en Google Colab.

## Preparación y decisiones
Se conservan variables numéricas, se eliminan duplicados exactos, se normaliza el texto del diagnóstico y se codifica `maligno=1`, `benigno=0`. No se imputan valores de forma previa porque no se identifican nulos en el archivo inicial; aun así, el pipeline incluye imputación por mediana como protección. El escalado se ajusta dentro del pipeline de regresión logística para evitar fuga de información; Random Forest no requiere estandarización.

## Evaluación
Partición de prueba estratificada 80/20 con `random_state=42`; validación cruzada `StratifiedKFold` de 5 folds; ajuste mediante `GridSearchCV` y selección por F1 macro. Se registran accuracy, precision macro, recall macro, F1 macro, ROC-AUC y latencia aproximada por fila. Los resultados de referencia se encuentran en `resultados/metricas_modelos.csv`.

## MLflow y reproducibilidad
MLflow guarda parámetros, métricas, modelos y archivos de resultados por ejecución. La fuente de `fuentes/train.py` crea un experimento local. Al ejecutar el notebook en Colab se crea la carpeta `mlruns` y la tabla exportable `resultados/mlflow_runs.csv`. La latencia depende del hardware y debe compararse en el mismo entorno.

## Limitaciones y conclusión
El conjunto contiene datos históricos y biomédicos; no debe usarse para diagnosticar personas. La comparación es didáctica, depende de una división y validación específicas, y el contraste de Wilcoxon con cinco folds es exploratorio. Antes de cualquier uso real se requiere validación clínica, revisión de sesgos, privacidad, seguridad y un proceso formal de gobernanza.
