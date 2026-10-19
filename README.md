Actividad 5: Clasificación de cáncer de mama

Descripción del proyecto

El objetivo es preparar un dataset público y comparar dos modelos de clasificación: regresión logística y Random Forest, utilizando Python y MLflow para registrar los experimentos.

Dataset

Se utiliza el dataset público Wisconsin Breast Cancer Diagnostic, con 569 registros y 30 variables predictoras. Los datos se dividen en una versión original y otra limpia.

Modelos

Regresión logística.

Random Forest.

Se utiliza validación cruzada de cinco grupos y búsqueda de hiperparámetros mediante GridSearchCV.

Entorno de ejecución

Python, scikit-learn, pandas, matplotlib y MLflow. El notebook está preparado para ejecutarse en Google Colab.

Resultados

En la ejecución realizada, la regresión logística obtuvo una exactitud del 97,37 %, una precisión macro del 97,41 %, un F1 macro del 97,16 % y un ROC-AUC del 99,57 %. Random Forest obtuvo una exactitud del 94,74 %, una precisión macro del 94,35 %, un F1 macro del 94,35 % y un ROC-AUC del 99,42 %.

En esta ejecución, la regresión logística obtuvo mejores resultados en las métricas evaluadas. Sin embargo, el proyecto es académico y el modelo no debe utilizarse para diagnosticar pacientes reales.

Cómo ejecutar

Abrir fuentes/entrena.ipynb en Google Colab.

Instalar las dependencias indicadas en requirements.txt.

Ejecutar las celdas en orden.

Consultar las métricas, gráficas y corridas registradas en MLflow.

Limitaciones

Este proyecto es académico. Los resultados no deben utilizarse para diagnosticar pacientes reales y se requiere validación adicional antes de cualquier aplicación clínica.
