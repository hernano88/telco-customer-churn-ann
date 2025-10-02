# Proyecto 4. Customer Churn Prediction (Deep Learning - ANN)

Modelo de red neuronal artificial para predecir la baja de clientes en telecomunicaciones (churn), utilizando el dataset de Telco Customer Churn. Incluye limpieza, transformación de variables categóricas, escalado, entrenamiento de red neuronal y evaluación con métricas de clasificación.

---

## 🎯 Objetivo

- Predecir la probabilidad de que un cliente abandone el servicio (churn).

- Transformar variables categóricas y numéricas en un dataset listo para modelado.

- Entrenar y evaluar una red neuronal artificial (ANN) para clasificación binaria.

---

## 🛠️ Desarrollo

- Preprocesamiento de datos

Eliminación de customerID.

Conversión de TotalCharges a numérico y manejo de nulos.

Recodificación de variables binarias “Yes/No” a 1/0.

One-hot encoding de variables categóricas (InternetService, Contract, PaymentMethod).

Escalado de variables continuas (tenure, MonthlyCharges, TotalCharges) con MinMaxScaler.

- Modelado

Split 80/20 en train/test.

Red neuronal con Keras:

Capa densa 26 neuronas (ReLU).

Capa oculta 15 neuronas (ReLU).

Capa de salida 1 neurona (Sigmoid).

Entrenamiento durante 100 épocas con adam y binary_crossentropy.

- Evaluación

Reporte de clasificación (precision, recall, f1-score).

Matriz de confusión para identificar falsos positivos y negativos.

---

## 📸 Ejemplos (código del notebook)
Construcción del modelo
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

Evaluación
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))

sn.heatmap(cm, annot=True, fmt='d')

---

## 📊 Resultados

- Accuracy global: ~80%

- Precision (Clase 0 – No Churn): 84%

- Precision (Clase 1 – Churn): 61%

- Recall (Clase 0 – No Churn): 84%

- Recall (Clase 1 – Churn): 61%

👉 El modelo logra una buena capacidad para identificar clientes que se quedan (No Churn) y un recall aceptable en la clase Churn, útil para detectar clientes con riesgo de abandono.

---

## 🔧 Tecnologías utilizadas

- Python

- Pandas / NumPy (procesamiento de datos)

- Matplotlib / Seaborn (visualización)

- Scikit-learn (preprocesamiento, métricas)

- TensorFlow / Keras (red neuronal artificial)
