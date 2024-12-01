"""
Este módulo se encarga de cargar y mostrar los datos del archivo CSV 'smartgrid.csv'.
Utiliza la librería pandas para leer el archivo y mostrar las primeras filas del dataset.
"""

import pandas as pd

datasmartgrid = pd.read_csv('../data/raw/smartgrid.csv')
datasmartgrid.head()
