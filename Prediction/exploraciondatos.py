"""
Este módulo se tiene para la exploración de datos.
"""

import pandas as pd

datasmartgrid = pd.read_csv('../data/raw/smartgrid.csv')

# Mostrar la información general
print(datasmartgrid.info())

# Estadísticas de las columnas
print(datasmartgrid.describe())

# Verificación de nulos
col_con_nan = []
for col in datasmartgrid.columns:
    porcentaje_faltante = datasmartgrid[col].isnull().mean()
    if porcentaje_faltante > 0:
        col_con_nan.append(col)
print(col_con_nan)

# Verificación de columnas categorícas, continuas y discretas
categoricas = [
    col for col in datasmartgrid.columns if datasmartgrid[col].dtypes == 'object'
]
print(categoricas)

continuas = [
    col for col in datasmartgrid.columns
    if datasmartgrid[col].dtypes in ['int64', 'float64'] and len(datasmartgrid[col].unique()) > 30
]
print(continuas)

discretas = [
    col for col in datasmartgrid.columns
    if datasmartgrid[col].dtypes in ['int64', 'float64'] and len(datasmartgrid[col].unique()) <= 30
]
print(discretas)

# Mostrar la cantidad de valores en 'stabf'
print(datasmartgrid['stabf'].value_counts())

# Mostrar la frecuencia normalizada
print(datasmartgrid['stabf'].value_counts(normalize=True) * 100)