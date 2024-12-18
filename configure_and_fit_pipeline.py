"""
Este módulo procesa el mejor modelo a utilizar.
"""

import pickle
import warnings
import configparser

import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.exceptions import ConvergenceWarning

# Ignorar advertencias de convergencia
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def configure_and_train_pipeline():
    """
    Fórmula del procesamiento el mejor modelo a utilizar.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('Smart Grid Predict Model')
    project_path = os.getcwd()

    # Cargar el pipeline previamente guardado
    with open(os.path.join(project_path, "artifacts", "pipeline.pkl"), 'rb') as f:
        smartgrid_pipeline = pickle.load(f)

    # Cargar los datos de entrenamiento y prueba
    train_data = pd.read_csv(os.path.join(project_path,
        'data', 'processed', 'feautures_for_models.csv'))
    test_data = pd.read_csv(os.path.join(project_path,
        'data', 'processed', 'test_dataset.csv'))

    # Leer la configuración
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    # Preparar las características (features) y la variable objetivo (target)
    x_features = train_data.drop(
        labels=list(config.get('GENERAL', 'VARS_TO_DROP').split(', ')),
        axis=1
    )
    y_target = train_data[config.get('GENERAL', 'TARGET')]

    x_features_test = test_data.drop(
        labels=list(config.get('GENERAL', 'VARS_TO_DROP').split(', ')),
        axis=1
    )
    y_target_test = test_data[config.get('GENERAL', 'TARGET')]

    with mlflow.start_run():
        modelos = {
            # KNN models
            'KNN_default': KNeighborsClassifier(),
            'KNN_optimized': KNeighborsClassifier(
                n_neighbors=5, weights='distance', algorithm='auto'),
            'KNN_optimized2': KNeighborsClassifier(
                n_neighbors=10, weights='uniform', algorithm='ball_tree'),

            # DecisionTree models
            'DecisionTree_default': DecisionTreeClassifier(),
            'DecisionTree_optimized': DecisionTreeClassifier(
                max_depth=10, min_samples_split=5),

            # RandomForest models
            'RandomForest_default': RandomForestClassifier(),
            'RandomForest_optimized': RandomForestClassifier(
                n_estimators=150, max_depth=20),

            # GradientBoosting models
            'GradientBoosting_default': GradientBoostingClassifier(),
            'GradientBoosting_optimized': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05),

            # LogisticRegression models
            'LogisticRegression_default': LogisticRegression(),
            'LogisticRegression_optimized': LogisticRegression(
                C=0.5, max_iter=1000),
            'LogisticRegression_optimized2': LogisticRegression(
                penalty='l2', C=1.0, solver='liblinear'),

            # XGBoost models
            'XGBoost_default': XGBClassifier(),
            'XGBoost_optimized': XGBClassifier(
                n_estimators=300, learning_rate=0.1),
            'XGBoost_optimized2': XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05)
        }

        resultados = {}
        for nombre, modelo in modelos.items():
            modelo.fit(x_features, y_target)
            y_preds = modelo.predict(x_features_test)
            acc = accuracy_score(y_target_test, y_preds)
            resultados[nombre] = acc
            print(f'{nombre} Accuracy: {acc}')

            mlflow.log_metric(f'{nombre}_accuracy', acc)
            mlflow.sklearn.log_model(modelo, f'{nombre}_model')
        mlflow.end_run()

    # Evaluar los modelos
    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(x_features, y_target)
        y_preds = modelo.predict(x_features_test)
        acc = accuracy_score(y_target_test, y_preds)
        resultados[nombre] = acc

    # Encontrar el mejor modelo
    mejor_modelo_nombre = max(resultados, key=resultados.get)
    mejor_modelo = modelos[mejor_modelo_nombre]

    # Añadir el mejor modelo al pipeline
    smartgrid_pipeline.steps.append((
        f'modelo_{mejor_modelo_nombre}', mejor_modelo))

    # Guardar el pipeline final con el modelo elegido
    with open(os.path.join(project_path,
        'artifacts', 'pipeline_final.pkl'), 'wb') as f:
        pickle.dump(smartgrid_pipeline, f)

configure_and_train_pipeline()
