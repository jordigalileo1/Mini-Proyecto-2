"""
Este módulo procesa un dataset para crear un pipeline de procesamiento de datos,
y guarda las características procesadas y el pipeline entrenado.
"""

import os
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def create_features():
    """
    Crea un pipeline para procesar características de un dataset, 
    divide los datos en conjuntos de entrenamiento y prueba, 
    y guarda los resultados en archivos.
    """
    
    # Leer el archivo de datos y la configuración
    dataset = pd.read_csv(os.path.join(os.getcwd(), 'data', 'raw', 'smartgrid.csv'))
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), 'pipeline.cfg'))

    # Preparación de las características y el objetivo
    x_features = dataset.drop(labels=list
                              (config.get('GENERAL', 'VARS_TO_DROP').split(', ')), axis=1)
    y_target = dataset[config.get('GENERAL', 'TARGET')]
    y_target = LabelEncoder().fit_transform(y_target)

    # Dividir en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.2, shuffle=True, random_state=2025)

    # Crear y entrenar el pipeline
    smartgrid_pipeline = Pipeline([
        ('feature_scaling', StandardScaler())
    ])
    smartgrid_pipeline.fit(x_train)

    # Procesar características de entrenamiento
    x_features_processed = smartgrid_pipeline.transform(x_train)
    df_features_processed = pd.DataFrame(x_features_processed, columns=x_train.columns)
    df_features_processed['stabf'] = y_train
    df_features_processed.to_csv(os.path.join(os.getcwd(),
        'data', 'processed', 'features_for_models.csv'), index=False)

    # Procesar características de prueba
    x_features_processed_test = smartgrid_pipeline.transform(x_test)
    df_features_processed_test = pd.DataFrame(x_features_processed_test, columns=x_test.columns)
    df_features_processed_test['stabf'] = y_test
    df_features_processed_test.to_csv(os.path.join(os.getcwd(),
        'data', 'processed', 'test_dataset.csv'), index=False)

    # Guardar el pipeline entrenado en un archivo
    with open(os.path.join(os.getcwd(), 'artifacts', 'pipeline.pkl'), 'wb') as f:
        pickle.dump(smartgrid_pipeline, f)

create_features()
