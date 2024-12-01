"""
Este modulo contiene las funciones para aplicar la ingenieria de caracteristicas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def create_miodel_fatures():
    """
    Crea un conjunto de características a partir del archivo raw 'smartgrid.csv',
    analiza outliers y guarda los datos preparados en un archivo CSV procesado.

    Lee los datos, identifica variables continuas y discretas, analiza outliers 
    y transforma la variable 'stabf' en una variable binaria.

    Args:
        None

    Returns:
        None. Genera un archivo CSV con datos procesados.
    """
    datasmartgrid = pd.read_csv('../data/raw/smartgrid.csv')
    datasmartgrid.head()

    dataset_proy = datasmartgrid
    dataset_proy.head()

    def get_variables_scale(dataset):
        """
        Identifica variables continuas y discretas en un DataFrame.

        Args:
            dataset (pd.DataFrame): Conjunto de datos a analizar.

        Returns:
            tuple: Una tupla con dos listas: (continuas, discretas), donde:
                - continuas (list): Columnas continuas
                (float64/int64 con más de 30 valores únicos).
                - discretas (list): Columnas discretas
                (float64/int64 con 30 o menos valores únicos).
        """
        continuas = [col for col in dataset.columns
                     if dataset[col].dtype in ['float64', 'int64']
                     and len(dataset[col].unique()) > 30]
        discretas = [col for col in dataset.columns
                     if dataset[col].dtype in ['float64', 'int64']
                     and len(dataset[col].unique()) <= 30]

        return continuas, discretas

    cont, disct = get_variables_scale(dataset_proy)

    # Creamos un DataFrame para las variables continuas
    proy_cont = pd.DataFrame(dataset_proy)
    cont, disct = get_variables_scale(proy_cont)
    df_continuas = proy_cont[cont]
    df_continuas.head()

    # Función para graficar las variables de la columna
    def plot_outliers_analysis(dataset, col):
        """
        Genera gráficos para el análisis de outliers de una columna específica.

        Args:
            dataset (pd.DataFrame): Conjunto de datos que contiene la columna.
            col (str): Nombre de la columna a analizar.

        Returns:
            None. Muestra gráficos de histograma, QQ-Plot y boxplot.
        """
        plt.figure(figsize=(10, 2))
        print(col)
        plt.subplot(131)
        dataset[col].hist(bins=50, density=True, color='red')
        plt.title("Densidad - Histograma")
        plt.subplot(132)
        stats.probplot(dataset[col], dist="norm", plot=plt)
        plt.title("QQ-Plot")
        plt.subplot(133)
        sns.boxplot(y=dataset[col])
        plt.title("Boxplot")
        plt.show()

    for col in cont:
        plot_outliers_analysis(proy_cont, col)

    # Reemplazar la variable 'stabf' por valores binarios
    datasmartgrid['stabf'] = datasmartgrid['stabf'].replace({'stable': 1, 'unstable': 0})

    # Guardar el DataFrame procesado en un archivo CSV
    datasmartgrid.to_csv('../data/processed/dataSmartPreparada.csv', index=False)
