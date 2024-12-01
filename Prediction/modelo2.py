"""
Script para entrenar y evaluar modelos de Árbol de Decisión utilizando diferentes configuraciones.
"""

# Importación de librerías
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configuración para ignorar advertencias
warnings.filterwarnings("ignore")


def create_model_features():
    """
    Función para crear y evaluar varios modelos de árboles 
    con diferentes configuraciones utilizando un conjunto de datos preparado.
    """
    # Importación de datos
    datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')

    # Creación de las variables x_features (características) e y_target (etiqueta objetivo)
    x_features = datasmartgrid.drop('stabf', axis=1)
    y_target = datasmartgrid['stabf']

    # División de los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.2, random_state=42
    )

    # Modelo 1: Árbol de Decisión básico
    model_basic = DecisionTreeClassifier(random_state=42)
    model_basic.fit(x_train, y_train)
    y_pred_basic = model_basic.predict(x_test)
    accuracy_basic = accuracy_score(y_test, y_pred_basic)

    # Modelo 2: Árbol de Decisión con profundidad máxima limitada
    model_depth_limited = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_depth_limited.fit(x_train, y_train)
    y_pred_depth_limited = model_depth_limited.predict(x_test)
    accuracy_depth_limited = accuracy_score(y_test, y_pred_depth_limited)

    # Modelo 3: Árbol de Decisión con control sobre el sobreajuste
    model_regularized = DecisionTreeClassifier(
        min_samples_split=10, min_samples_leaf=5, random_state=42
    )
    model_regularized.fit(x_train, y_train)
    y_pred_regularized = model_regularized.predict(x_test)
    accuracy_regularized = accuracy_score(y_test, y_pred_regularized)

    # Modelo 4: Árbol de Decisión con control de complejidad adicional
    model_complex = DecisionTreeClassifier(
        criterion='entropy', min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    model_complex.fit(x_train, y_train)
    y_pred_complex = model_complex.predict(x_test)
    accuracy_complex = accuracy_score(y_test, y_pred_complex)

    # Mostrar resultados
    print(f"Modelo 1 (básico) - Precisión: {accuracy_basic:.4f}")
    print(f"Modelo 2 (max_depth=5) - Precisión: {accuracy_depth_limited:.4f}")
    print(f"Modelo 3 (min_samples_split=10, min_samples_leaf=5) - Precisión: {accuracy_regularized:.4f}")
    print(f"Modelo 4 (criterio='entropy', min_samples_split=20, min_samples_leaf=10) - Precisión: {accuracy_complex:.4f}")