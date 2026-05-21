# COMPENDIO DE MODELOS DE MACHINE LEARNING Y DEEP LEARNING

---

## I. MODELOS DE REGRESIÓN

### Regresión Lineal

**Tipo:** Supervisado - Regresión.

**Cuándo usarlo:** Cuando existe una relación lineal clara entre las variables independientes y el target. No usar si los residuos muestran patrones no aleatorios (como parábolas).

**Características clave:** Minimiza el error cuadrático medio (MSE). Es altamente interpretable ("caja blanca"). Sensible a outliers.

**Hiperparámetros principales:** `fit_intercept` (si se calcula el sesgo).

**Ejemplo de código (Datos de Ventas):**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
```

---

### Modelos Regularizados (Ridge, Lasso, Elastic Net)

**Tipo:** Supervisado - Regresión con penalización.

**Cuándo usarlo:** Cuando hay muchas features (posible sobreajuste) o alta correlación entre ellas. Lasso es ideal si queremos hacer selección automática de variables.

**Características clave:** Ridge (L2) evita pesos extremos; Lasso (L1) puede llevar pesos a cero; Elastic Net combina ambos.

**Hiperparámetros principales:** `alpha` (fuerza de regularización) y `l1_ratio` (en Elastic Net).

**Ejemplo de código:**
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01, max_iter=10000)
model.fit(X_train_scaled, y_train)
```

---

## II. MODELOS DE CLASIFICACIÓN

### Regresión Logística

**Tipo:** Supervisado - Clasificación binaria o multiclase.

**Cuándo usarlo:** Para predecir probabilidades de pertenencia a una clase (ej. fraude, impago). No usar si la relación es puramente no lineal.

**Características clave:** Utiliza la función sigmoide para acotar el resultado entre 0 y 1. Asume independencia de observaciones.

**Hiperparámetros principales:** `C` (inversa de regularización) y `penalty` ('l1', 'l2').

**Ejemplo de código:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000, C=1.0)
model.fit(X_train, y_train)
```

---

### K-Nearest Neighbors (KNN)

**Tipo:** Supervisado - Clasificación / Regresión.

**Cuándo usarlo:** Datasets pequeños o cuando no se quieren hacer suposiciones sobre la distribución. No usar en datasets masivos (lento en predicción).

**Características clave:** "Lazy learner" (no entrena, solo guarda datos). Se basa en distancias euclídeas; requiere escalado previo de datos.

**Hiperparámetros principales:** `n_neighbors` (número de vecinos; k bajo produce overfitting).

**Ejemplo de código:**
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
```

---

### Support Vector Machines (SVM)

**Tipo:** Supervisado - Clasificación (SVC) / Regresión (SVR).

**Cuándo usarlo:** Problemas con muchas features o fronteras de decisión complejas mediante kernels. No usar si el número de features es mucho mayor al de muestras.

**Características clave:** Busca el hiperplano que maximiza el margen entre clases. El "kernel trick" permite modelar relaciones no lineales sin aumentar explícitamente las dimensiones.

**Hiperparámetros principales:** `C` (tolerancia a errores), `kernel` ('linear', 'rbf', 'poly') y `gamma` (influencia de puntos lejanos).

**Ejemplo de código:**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)
```

---

### Árboles de Decisión y Random Forest

**Tipo:** Supervisado - Clasificación / Regresión.

**Cuándo usarlo:** Cuando se busca interpretabilidad (árbol simple) o alta robustez frente a outliers (Random Forest).

**Características clave:** Dividen el espacio según condiciones if-else. Random Forest es un ensemble de muchos árboles entrenados con bootstrapping (bagging), reduciendo la varianza.

**Hiperparámetros principales:** `max_depth` (profundidad máxima), `n_estimators` (número de árboles) y `max_features`.

**Ejemplo de código:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
```

---

## III. MODELOS NO SUPERVISADOS

### K-Means (Clustering)

**Tipo:** No supervisado - Agrupamiento.

**Cuándo usarlo:** Segmentación de clientes, compresión de imágenes o detección de anomalías.

**Características clave:** Agrupa datos en k grupos minimizando la distancia a los centroides. No sabe qué es cada clase, solo detecta patrones.

**Hiperparámetros principales:** `n_clusters` (k).

**Ejemplo de código (Compresión de imagen):**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X_img_reshaped)
```

---

### PCA (Análisis de Componentes Principales)

**Tipo:** No supervisado - Reducción de dimensionalidad.

**Cuándo usarlo:** Visualización de datos de alta dimensión, eliminación de ruido o preprocesado antes de otros modelos.

**Características clave:** Transforma variables correlacionadas en componentes no correlacionados que capturan la máxima varianza.

**Hiperparámetros principales:** `n_components`.

**Ejemplo de código:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
```

---

## IV. DEEP LEARNING (REDES NEURONALES)

### Red Neuronal Artificial (ANN / MLP)

**Tipo:** Supervisado - Clasificación / Regresión.

**Cuándo usarlo:** Problemas tabulares complejos o datos no lineales masivos. No usar si hay muy pocos datos.

**Características clave:** Simulan neuronas biológicas con capas de entrada, ocultas y salida. Utilizan backpropagation y descenso de gradiente para aprender pesos.

**Hiperparámetros principales:** `hidden_layer_sizes`, funciones de activación (`relu`, `softmax`, `sigmoid`) y optimizadores (`adam`, `sgd`).

**Ejemplo de código (Keras):**
```python
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)
```

---

### Redes Neuronales Convolucionales (CNN)

**Tipo:** Supervisado - Principalmente Clasificación de Imágenes.

**Cuándo usarlo:** Visión artificial, reconocimiento de objetos y detección de patrones espaciales.

**Características clave:** Capas convolucionales para extraer características y capas de pooling para reducir dimensiones.

**Hiperparámetros principales:** Número de filtros, tamaño del kernel, dropout.

**Ejemplo de código (Clasificador de Paisajes):**
```python
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48,48,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])
```

---

### Redes Neuronales Recurrentes (RNN / LSTM)

**Tipo:** Supervisado - Series Temporales / NLP.

**Cuándo usarlo:** Datos secuenciales (texto, audio, series temporales) donde el orden importa.

**Características clave:** Tienen "memoria" de estados anteriores. Las capas LSTM (Long Short-Term Memory) evitan que la información se pierda en secuencias largas.

**Hiperparámetros principales:** `units` (neuronas en la capa recurrente).

**Ejemplo de código (Predicción de Serie Temporal):**
```python
from keras.layers import LSTM, Dense
model = keras.Sequential([
    LSTM(units=128, input_shape=(1, 4), activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])
```

---

## V. TABLA COMPARATIVA RESUMEN

| Modelo | Tipo | Problema Principal | Fortalezas |
|---|---|---|---|
| Regresión Lineal | Supervisado | Regresión | Simple, interpretable, rápido |
| Logística | Supervisado | Clasificación | Eficiente para probabilidades, rápido |
| KNN | Supervisado | Ambos | No asume formas en los datos, flexible |
| SVM | Supervisado | Ambos | Versátil con kernels, eficaz con muchas features |
| Árbol Decisión | Supervisado | Ambos | Intuitivo, no requiere escalado de datos |
| Random Forest | Supervisado | Ambos | Robusto, reduce overfitting, muy preciso |
| Boosting (XGB) | Supervisado | Ambos | Máxima precisión competitiva |
| K-Means | No Supervisado | Clustering | Segmentación automática y sencilla |
| PCA | No Supervisado | Red. Dimens. | Reduce ruido y facilita visualización |
| ANN (Deep L.) | Supervisado | Ambos | Potente para datos tabulares masivos complejos |
| CNN | Supervisado | Clasificación | El estándar para imágenes y visión artificial |
| LSTM (RNN) | Supervisado | Series Temp. | Excelente para secuencias y memoria temporal |
