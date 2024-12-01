"""
Este módulo entrena un modelo de Random Forest utilizando datos procesados
y evalúa el rendimiento con un reporte de clasificación.
"""

# Importaciones estándar
import warnings
warnings.filterwarnings("ignore")

# Importaciones de terceros
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Importación de data
datasmartgrid = pd.read_csv('../data/processed/dataSmartPreparada.csv')
datasmartgrid.head()

# Creación de la variable X
X = datasmartgrid.drop('stabf', axis=1)

# Creación de la variable y
y = pd.DataFrame(datasmartgrid['stabf'])

# Creación de los Sets de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear un diccionario con configuraciones de hiperparámetros
configuraciones = [
    {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'random_state': 42},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10, 'random_state': 42}
]

# Iterar sobre las configuraciones
for i, config in enumerate(configuraciones, 1):
    print(f"Configuración {i}: {config}")

    # Crear el modelo con la configuración actual
    model = RandomForestClassifier(**config)

    # Entrenar el modelo
    model.fit(X_train, y_train.values.ravel())

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    print(f"Reporte de clasificación para Configuración {i}:\n")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
