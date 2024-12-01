"""
Este script entrena y evalúa un modelo SVM utilizando diferentes configuraciones de hiperparámetros
sobre un conjunto de datos relacionado con la estabilidad de un sistema. 
Se utilizan métricas como el reporte de clasificación para evaluar el rendimiento del modelo.
"""

# Importación de librerías
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Configuración de advertencias
warnings.filterwarnings("ignore")

def modelo4():
    """
    Función que entrena y evalúa un modelo SVM sobre un conjunto de datos relacionado con la estabilidad de un sistema.
    """
    # Importación de datos
    datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')
    datasmartgrid.head()

    # Creación de la variable x
    x = datasmartgrid.drop('stabf', axis=1)

    # Creación de la variable y
    y = pd.DataFrame(datasmartgrid['stabf'])

    # Creación de los sets de entrenamiento
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Crear un diccionario con configuraciones de hiperparámetros
    configuraciones = [
        {'kernel': 'linear', 'C': 1, 'random_state': 42},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.1, 'random_state': 42},
        {'kernel': 'poly', 'C': 1, 'degree': 3, 'random_state': 42}
    ]

    # Iterar sobre las configuraciones
    for i, config in enumerate(configuraciones, 1):
        print(f"Configuración {i}: {config}")

        # Crear el modelo con la configuración actual
        model = SVC(**config)

        # Entrenar el modelo
        model.fit(x_train, y_train.values.ravel())

        # Realizar predicciones
        y_pred = model.predict(x_test)

        # Evaluar el modelo
        print(f"Reporte de clasificación para Configuración {i}:\n")
        print(classification_report(y_test, y_pred))
        print("-" * 50)

