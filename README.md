## Sistema de Detección de Fraude en Transacciones Financieras
# Descripción
Este proyecto implementa un modelo de Machine Learning para detectar transacciones fraudulentas utilizando un árbol de decisión (Decision Tree Classifier). El sistema analiza patrones en datos financieros para identificar operaciones sospechosas basándose en características como el tipo de transacción, monto y balances de cuenta.
Características

## Análisis exploratorio de datos con visualizaciones interactivas
- Preprocesamiento automático de variables categóricas
- Modelo de clasificación basado en Decision Tree
- Métricas de evaluación completas (precisión, recall, F1-score)
- Visualización de distribución de transacciones con gráficos de pastel

## Requisitos
pandas
numpy
plotly
scikit-learn

## Instalación
pip install pandas numpy plotly scikit-learn

## Datos

Este proyecto requiere un archivo datos_financieros.csv que no está incluido en el repositorio debido a su tamaño.

**Opciones para obtener los datos:**
- Descargar desde: https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download
- Contactar al autor para acceso a los datos

## Estructura de Datos
El archivo CSV (datos_financieros.csv) contiene las siguientes columnas:

type: Tipo de transacción (CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT)
amount: Monto de la transacción
oldbalanceOrg: Balance anterior de la cuenta origen
newbalanceOrig: Nuevo balance de la cuenta origen
isFraud: Etiqueta binaria (0 = Sin fraude, 1 = Fraude)

## Uso

Asegurarse de tener el archivo datos_financieros.csv en el mismo directorio que el script
Ejecuta el script:

python fraud_detection.py

## El programa realizará las siguientes operaciones:

- Carga y exploración inicial de datos
- Detección de valores nulos
- Visualización de distribución de tipos de transacciones
- Análisis de correlaciones
- Preprocesamiento de variables
- Entrenamiento del modelo
- Evaluación con métricas de desempeño

## Flujo del Algoritmo
# 1. Carga y Exploración
python- Carga del dataset
- Visualización de primeras filas
- Identificación de valores nulos
- Análisis de distribución de transacciones
  
# 2. Análisis de Correlación
Calcula correlaciones entre variables numéricas y la variable objetivo (isFraud)

# 4. Preprocesamiento

Normalización de tipos de transacción (mayúsculas y eliminación de espacios)
Codificación de tipos de transacción:

CASH_OUT: 1
PAYMENT: 2
CASH_IN: 3
TRANSFER: 4
DEBIT: 5


Transformación de etiquetas de fraude a texto descriptivo

# 4. Entrenamiento del Modelo

División de datos: 90% entrenamiento, 10% prueba
Algoritmo: Decision Tree Classifier
Features utilizadas: tipo, monto, balance anterior, balance nuevo

# 5. Evaluación
El modelo proporciona:

Accuracy: Precisión general del modelo
Precision: Exactitud de predicciones positivas
Recall: Capacidad de detectar fraudes reales
F1-Score: Media armónica entre precisión y recall
Matriz de Confusión: Visualización de predicciones correctas e incorrectas

## Salida del Programa
- Primeras filas del dataset
- Valores nulos por columna
- Gráfica interactiva de distribución
- Correlaciones con fraude
- Accuracy del modelo

# Métricas de evaluación:
  - Precisión
  - Recall
  - F1-score
  - Matriz de confusión

## Interpretación de Resultados

Precisión alta: El modelo identifica correctamente las transacciones fraudulentas
Recall alto: El modelo detecta la mayoría de los fraudes reales
F1-score: Balance entre precisión y recall
Matriz de confusión: Muestra verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos

## Limitaciones

El modelo utiliza solo 4 características para la predicción
Decision Tree puede sufrir de overfitting con datos complejos
Requiere balanceo de clases si hay desbalance significativo entre fraude/no fraude

