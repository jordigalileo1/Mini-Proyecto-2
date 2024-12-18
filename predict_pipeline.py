"""
Este módulo ya predice los datos.
"""

import os
import pickle
import warnings
from datetime import datetime
import pandas as pd

# Suprime todos los UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def predict_pipeline():
    """
    Fórmiula del módulo ya predice los datos.
    """
    project_path = os.getcwd()

    # Cargar el pipeline entrenado
    with open(os.path.join(project_path, "artifacts", "pipeline_final.pkl"), 'rb') as f:
        smartgrid_pipeline = pickle.load(f)

    # Cargar los datos de prueba
    test_data = pd.read_csv(os.path.join(project_path, 'data', 'processed', 'test_dataset.csv'))

    # Separar las características (X) y la variable objetivo (Y)
    x_features_test = test_data.drop(labels=['stabf'], axis=1)

    # Realizar la predicción
    y_pred = smartgrid_pipeline.predict(x_features_test)

    # Directorio donde se guardarán las predicciones
    PREDICTIONS_DIR = os.path.join(os.getcwd(), 'data', 'predictions', 'predict_dataset')

    # Crear el directorio si no existe
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)

    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    PREDICTIONS_FILE = f'{PREDICTIONS_DIR}/{timestamp}.csv'

    # Guardar las predicciones en un archivo CSV
    predictions_df = pd.DataFrame(y_pred, columns=['stabf_prediction'])
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

predict_pipeline()
