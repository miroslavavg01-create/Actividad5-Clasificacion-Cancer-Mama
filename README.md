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

En la ejecución realizada, la regresión logística obtuvo un accuracy de 97.37%, un F1 macro de 97.16% y un ROC-AUC de 99.57%. Random Forest obtuvo un accuracy de 94.74%, un F1 macro de 94.35% y un ROC-AUC de 99.42%.

Cómo ejecutar

Abrir fuentes/entrena.ipynb en Google Colab.

Instalar las dependencias indicadas en requirements.txt.

Ejecutar las celdas en orden.

Consultar las métricas, gráficas y corridas registradas en MLflow.

Limitaciones

Este proyecto es académico. Los resultados no deben utilizarse para diagnosticar pacientes reales y se requiere validación adicional antes de cualquier aplicación clínica.
