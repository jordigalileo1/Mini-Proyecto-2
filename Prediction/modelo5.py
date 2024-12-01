"""
Este módulo implementa el modelo K-Nearest Neighbors (KNN) para clasificación
en el conjunto de datos datasmartgrid, utilizando diferentes configuraciones de
hiperparámetros.
"""

# %% [markdown]
# # Modelos K-Nearest Neighbors (KNN)

# %% [python]
# Importación de librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %% [python]
datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')
datasmartgrid.head()

# %% [python]
# Separar características (X) y etiqueta (y)
X = datasmartgrid.drop(columns=['stabf'])  # Eliminar la columna objetivo
y = datasmartgrid['stabf']  # Columna objetivo

# %% [python]
# Dividir el dataset en entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [python]
# Escalador estándar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [python]
# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [python]
# Configuración hiper-parámetro 1: k=3, distancia euclidiana, ponderación uniforme
knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='uniform')
knn1.fit(X_train_scaled, y_train)
y_pred1 = knn1.predict(X_test_scaled)
print("Configuración 1: k=3, euclidiana, uniforme")
print(confusion_matrix(y_test, y_pred1))
#print(classification_report(y_test, y_pred1))

# %% [python]
# Configuración hiper-parámetro 2: k=5, distancia manhattan, ponderación por distancia
knn2 = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn2.fit(X_train_scaled, y_train)
y_pred2 = knn2.predict(X_test_scaled)
print("Configuración 2: k=5, manhattan, ponderación por distancia")
print(confusion_matrix(y_test, y_pred2))
#print(classification_report(y_test, y_pred2))

# %% [python]
# Configuración hiper-parámetro 3: k=7, distancia minkowski (p=3), ponderación uniforme
knn3 = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=3, weights='uniform')
knn3.fit(X_train_scaled, y_train)
y_pred3 = knn3.predict(X_test_scaled)
print("Configuración 3: k=7, minkowski (p=3), uniforme")
print(confusion_matrix(y_test, y_pred3))
#print(classification_report(y_test, y_pred3))

# %% [markdown]
# Configuración 1 (k=3, euclidiana, uniforme):

# *Verdaderos positivos: 7360
# *Falsos positivos: 365
# *Verdaderos negativos: 3957
# *Falsos negativos: 318

# Configuración 2 (k=5, manhattan, ponderación por distancia):

# *Verdaderos positivos: 7503
# *Falsos positivos: 217
# *Verdaderos negativos: 4105
# *Falsos negativos: 175

# Configuración 3 (k=7, minkowski p=3, uniforme):

# *Verdaderos positivos: 7455
# *Falsos positivos: 405
# *Verdaderos negativos: 3917
# *Falsos negativos: 223

# Conclusión:
# Configuración 2 (k=5, manhattan, ponderación por distancia) parece ser la mejor opción:

# --Tiene el menor número de falsos positivos (217) y falsos negativos (175),
# lo que implica que el modelo está clasificando correctamente más casos.
# Presenta un mejor balance entre sensibilidad y precisión.

# %% [python]
# Matriz de confusión para la Configuración 2
cm = confusion_matrix(y_test, y_pred2)

# Reporte de clasificación para Configuración 2
report = classification_report(y_test, y_pred2, output_dict=True)

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
             xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
plt.title("Matriz de Confusión - Configuración 2 (k=5, manhattan, ponderación por distancia)")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# Mostramos métricas clave
print("Reporte de Clasificación - Configuración 2")
print(classification_report(y_test, y_pred2))
