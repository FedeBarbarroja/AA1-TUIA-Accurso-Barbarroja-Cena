# AA1-TUIA-2025-Accurso-Barbarroja-Cena
## Predicción de Lluvia en Australia

Este trabajo práctico tiene como objetivo desarrollar un modelo de clasificación binaria para predecir si lloverá al día siguiente en Australia, utilizando un dataset histórico con variables meteorológicas (presión, temperatura, humedad, vientos, etc.).

## Pasos realizados

### Exploración de Datos (EDA) e Ingeniería de Características
* **Clusterización:** Agrupación de ciudades mediante **K-Means** para crear regiones climáticas y simplificar la variable geográfica.
* **Análisis de Desbalance:** Identificación de la desproporción de clases ("No Llueve" > "Sí Llueve") y decisión estratégica de métricas.

### Preprocesamiento Avanzado
* **Limpieza de Datos:** Eliminación de variables con excesivos valores nulos o riesgo de *data leakage*.
* **Imputación:** Implementación de **IterativeImputer** (imputación multivariada) para preservar las correlaciones entre variables meteorológicas.
* **Optimización de Memoria:** Manejo de **Matrices Dispersas** (`sparse_output=True`) en el One-Hot Encoding para evitar desbordamiento de RAM.
* **Escalado:** Estandarización de variables numéricas con `StandardScaler`.

### Modelado y Evaluación
* **Definición de Métricas:** Selección de **AUC-ROC** como métrica principal por ser robusta al desbalance e independiente del umbral.
* **Modelo Base:** Implementación de Regresión Logística con `class_weight='balanced'`.
* **Modelos Avanzados:**
    * **AutoML (PyCaret):** Entrenamiento y diagnóstico de modelos basados en árboles (LightGBM). Detección de *overfitting*.
    * **Redes Neuronales (TensorFlow/Keras):** Diseño de una arquitectura densa con *Dropout* y *EarlyStopping*.

### Optimización
* **Ajuste de Hiperparámetros:** Uso de `RandomizedSearchCV` y `Optuna` para optimizar la Regresión Logística y la arquitectura de la Red Neuronal.

### Explicabilidad
* **SHAP (SHapley Additive exPlanations):** Análisis de importancia de variables para abrir la "Blackbox" de la Red Neuronal y validar la coherencia física del modelo.

### MLOps y Despliegue
* **Serialización:** Guardado del pipeline de preprocesamiento (`joblib`) y el modelo (`.h5`).
* **Docker:** Encapsulamiento del modelo y script de inferencia en un contenedor Docker para asegurar reproducibilidad en producción.

## Archivos a Utilizar:
* **TP-clasificacion-AA1_Entrega_3.ipynb:** Archivo que contiene todo el código desde el punto 1 hasta el 11
* **requirements.txt** Archivo con todas las librerías y versiones utilizadas.
* **Carpeta Docker** Carpeta con todos los archivos necesarios para la correcta construccion de la imagen del contenedor y su ejecución incluidas las instrucciones necesarias.


## Aprendizajes y Conclusiones
* **Complejidad vs. Linealidad:** La Red Neuronal superó a la Regresión Logística (AUC 0.90 vs 0.87), demostrando la necesidad de capturar patrones no lineales en datos climáticos.
* **Importancia del Preprocesamiento:** La ingeniería de características (manejo de fechas y regiones) fue crítica para la viabilidad técnica y el rendimiento del modelo.
* **Docker:** La importancia de "containerizar" el entorno para evitar problemas de dependencias y versiones entre desarrollo y producción.
