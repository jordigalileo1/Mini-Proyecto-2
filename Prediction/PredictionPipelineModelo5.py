# %% [markdown]
# # Prediction Pipeline
# Notebook para definir funciones de preprocesamiento, entrenamiento y predicción.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
# Función para cargar y dividir los datos
def load_and_split_data(filepath, target_column, test_size=0.2, random_state=42):
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# %%
# Función para preprocesar datos (escalado)
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# %%
# Función para entrenar un modelo KNN
def train_knn(X_train_scaled, y_train, n_neighbors=3, metric='euclidean', weights='uniform', p=2):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights, p=p)
    knn.fit(X_train_scaled, y_train)
    return knn

# %%
# Función para realizar predicciones
def predict(model, X_test_scaled):
    return model.predict(X_test_scaled)

# %%
# Función completa para ejecutar el pipeline
def prediction_pipeline(filepath, target_column):
    # Carga y división de datos
    X_train, X_test, y_train, y_test = load_and_split_data(filepath, target_column)

    # Preprocesamiento
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Entrenamiento
    model = train_knn(X_train_scaled, y_train)

    # Predicción
    y_pred = predict(model, X_test_scaled)
    
    return model, scaler, y_test, y_pred
