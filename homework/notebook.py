
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import json


train_df = pd.read_csv("files/input/train_data.csv.zip")
test_df = pd.read_csv("files/input/test_data.csv.zip")

#limpieza de datos
def prepare_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace(0,np.nan)
    df["EDUCATION"] = df["EDUCATION"].clip(upper=4)
    df["EDUCATION"] = df["EDUCATION"].map({1: "1", 2: "2", 3: "3", 4: "others"})
    df = df.drop_duplicates()
    df = df.dropna()
    return df

train_df = prepare_data(train_df)
test_df = prepare_data(test_df)


# ---------------------Paso 2.-----------------------------------------
# Divida los datasets en x_train, y_train, x_test, y_test.
#
x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]
x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]

# ---------------------Paso 3.-----------------------------------------
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (random forest).

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_features = [c for c in x_train.columns if c not in categorical_features]

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_features),
    ("num", numeric_pipeline, numeric_features)
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(random_state=42))
])


# ---------------------Paso 4.-----------------------------------------
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': [ 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

# ---------------------Paso 5.-----------------------------------------
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

import pickle, gzip
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

# ---------------------Paso 6.-----------------------------------------
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}


metrics = []

train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}
metrics.append(train_metrics)

test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
}
metrics.append(test_metrics)

with open('files/output/metrics.json', 'w') as f:
    for m in metrics:
        f.write(json.dumps(m) + '\n')


# ---------------------Paso 7.-----------------------------------------
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Matriz de confusión para entrenamiento
cm_train = confusion_matrix(y_train, y_train_pred)
cm_train_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
    'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
}
metrics.append(cm_train_dict)

# Matriz de confusión para prueba
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
    'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
}
metrics.append(cm_test_dict)

# Guardar todo en una sola escritura
with open('files/output/metrics.json', 'w') as f:
    for m in metrics:
        f.write(json.dumps(m) + '\n')