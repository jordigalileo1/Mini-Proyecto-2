"""
Este módulo implementa el modelo Xgboost para clasificación
en el conjunto de datos datasmartgrid, utilizando diferentes configuraciones de
hiperparámetros.
"""

# Importación de librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')
datasmartgrid.head()

# Separar características (X) y etiqueta (y)
X = datasmartgrid.drop(columns=['stabf'])  # Eliminar la columna objetivo
y = datasmartgrid['stabf']  # Columna objetivo

# Dividir el dataset en entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalador estándar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuración hiperparámetro 1: Usando XGBoost con configuración básica
xgb1 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb1.fit(X_train_scaled, y_train)
y_pred1 = xgb1.predict(X_test_scaled)

print("Configuración 1: XGBoost básico (100 estimadores, max_depth=3, learning_rate=0.1)")
print(confusion_matrix(y_test, y_pred1))

# Configuración hiperparámetro 2: XGBoost con mayor profundidad y regularización
xgb2 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, reg_alpha=0.1, random_state=42)
xgb2.fit(X_train_scaled, y_train)
y_pred2 = xgb2.predict(X_test_scaled)

print("Configuración 2: XGBoost con mayor profundidad y regularización")
print(confusion_matrix(y_test, y_pred2))

# Configuración hiperparámetro 3: XGBoost con ajuste fino
xgb3 = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, gamma=0.2, random_state=42)
xgb3.fit(X_train_scaled, y_train)
y_pred3 = xgb3.predict(X_test_scaled)

print("Configuración 3: XGBoost con ajuste fino")
print(confusion_matrix(y_test, y_pred3))

# Matriz de confusión para la Configuración 2
cm = confusion_matrix(y_test, y_pred2)

# Reporte de clasificación para Configuración 2
report = classification_report(y_test, y_pred2, output_dict=True)

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
plt.title("Matriz de Confusión - Configuración 2 (XGBoost)")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# Mostramos métricas clave
print("Reporte de Clasificación - Configuración 2")
print(classification_report(y_test, y_pred2))
