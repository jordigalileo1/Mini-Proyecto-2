"""
Este módulo implementa múltiples modelos de regresión logística para predecir la variable de estabilidad 
('stabf') usando un conjunto de datos procesado.
"""

# Importación de bibliotecas estándar y de terceros
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Configuración para ignorar advertencias
warnings.filterwarnings("ignore")


def create_model_features():
    """
    Función para crear y evaluar varios modelos de regresión logística 
    con diferentes configuraciones utilizando un conjunto de datos preparado.
    """
    # Importación de los datos
    datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')

    # Creación de las variables de características (X) y etiqueta (y)
    features = datasmartgrid.drop('stabf', axis=1)
    target = datasmartgrid['stabf']

    # Creación de los conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Modelo 1: Regresión Logística básica
    model_basic = LogisticRegression()
    model_basic.fit(x_train, y_train)
    y_pred_basic = model_basic.predict(x_test)
    accuracy_basic = accuracy_score(y_test, y_pred_basic)

    # Modelo 2: Regresión Logística con regularización L2 (Ridge)
    model_ridge = LogisticRegression(penalty='l2')
    model_ridge.fit(x_train, y_train)
    y_pred_ridge = model_ridge.predict(x_test)
    accuracy_ridge = accuracy_score(y_test, y_pred_ridge)

    # Modelo 3: Regresión Logística con regularización L1 (Lasso)
    model_lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    model_lasso.fit(x_train, y_train)
    y_pred_lasso = model_lasso.predict(x_test)
    accuracy_lasso = accuracy_score(y_test, y_pred_lasso)

    # Modelo 4: Regresión Logística con tolerancia y número máximo de iteraciones ajustados
    model_tuned = LogisticRegression(tol=1e-4, max_iter=200, random_state=42)
    model_tuned.fit(x_train, y_train)
    y_pred_tuned = model_tuned.predict(x_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

    # Mostrar resultados
    print(f"Modelo 1 (básico) - Precisión: {accuracy_basic:.4f}")
    print(f"Modelo 2 (penalización L2) - Precisión: {accuracy_ridge:.4f}")
    print(f"Modelo 3 (penalización L1) - Precisión: {accuracy_lasso:.4f}")
    print(f"Modelo 4 (ajustado) - Precisión: {accuracy_tuned:.4f}")