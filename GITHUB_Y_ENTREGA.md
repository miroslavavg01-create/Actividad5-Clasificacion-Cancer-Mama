# Publicación en GitHub y evidencia MLflow

## Subir la carpeta al repositorio
1. Crea un repositorio público o privado en GitHub.
2. Sube el contenido de `Actividad5_Supermercados_Miroslava` manteniendo las carpetas `datos`, `fuentes`, `resultados`, `evidencias` y `tests`.
3. Verifica que `README.md`, `CHANGELOG.md` y `requirements.txt` estén en la raíz del repositorio.

## Ejecutar y capturar MLflow
1. Abre `fuentes/entrena.ipynb` en Google Colab y ejecuta todas las celdas de arriba hacia abajo. La primera celda de código instala las dependencias.
2. Al finalizar, revisa la tabla de runs que aparece después del entrenamiento. Esa tabla sale de `mlflow.search_runs` y muestra los experimentos registrados.
3. Toma una captura de la tabla y, si es posible, una de la interfaz de MLflow Tracking. Guarda las capturas dentro de `evidencias/`.
4. Descarga `resultados_actividad5.zip` generado en el último paso. Incluye resultados, archivos de MLflow y la tabla de evidencia exportada.
5. Copia los archivos de resultados nuevos a `resultados/` y la evidencia a `evidencias/`, súbelos a GitHub y vuelve a crear el ZIP final para entregar.

No se incluye una captura ficticia del panel: las capturas deben salir de una ejecución real de MLflow.
