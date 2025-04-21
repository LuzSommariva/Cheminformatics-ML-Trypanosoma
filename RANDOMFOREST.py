
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc

import pickle

import pandas as pd
import os
import sys

#deepchem
from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset

print("¡Imports exitosos!")

# Verificamos si se pasó el nombre del archivo
if len(sys.argv) < 2:
    print("Uso: python prueba.py <nombre_archivo.csv>")
    sys.exit(1)

# Obtener el nombre del archivo desde los argumentos
filename = sys.argv[1]

# Ruta al archivo dentro de la carpeta "descriptores"
filepath = os.path.join("descriptores", filename)

# Verificamos si el archivo existe
if not os.path.exists(filepath):
    print(f"El archivo {filepath} no existe.")
    sys.exit(1)

# Cargar el archivo CSV
data_0= pd.read_csv(filepath)


#hacer splitting train,validation, test
def scaffold_dataset_splitter(df):
    df_target = df.copy()
    scaffoldsplitter = ScaffoldSplitter()
    smiles = df_target["smiles"].to_numpy()
    comp_id = df_target["comp_id"].to_numpy()
    label_bioact = df_target["bioactivity"].to_numpy()
    dc_dataset_split = NumpyDataset(X=smiles,y=label_bioact,ids=smiles)
    train,valid,test = scaffoldsplitter.train_valid_test_split(dc_dataset_split,frac_train=0.64,frac_valid=0.16,frac_test=0.20,seed=123)
    df_target.loc[df_target.smiles.isin(test.ids),"Data_split"] = "test"
    df_target.loc[df_target.smiles.isin(train.ids),"Data_split"] = "train"
    df_target.loc[df_target.smiles.isin(valid.ids),"Data_split"] = "validation"
    
    
    return df_target

data= scaffold_dataset_splitter(data_0)
# Mostramos las primeras filas
print(data.head())


# Crear directorios si no existen
os.makedirs("feature_importances_plots_RF", exist_ok=True)
os.makedirs("modelos_RF", exist_ok=True)
os.makedirs("resultados_RF", exist_ok=True)
os.makedirs("rocdata_RF", exist_ok=True)


#Dataframe para guardar los resultados on esas columnas
resultados_df = pd.DataFrame(columns=['Target', 'Compuestos Totales', 'Activos', 'Inactivos', 'Validation F1', 'Validation ROC AUC', 'Test F1', 'Test ROC AUC'])

#Lista de todos los targets que se utilizan
targets = data['target'].unique()

#Lista de los descriptores que se utilizan
descriptor_cols = [    "numb", "C", "O", "N", "Cl", "F", "Br", "I", "nh2_pattern",
    "ring5_counts", "ring6_counts", "MolWeight", "Rings",
    "TPSA", "FpDM1", "RotBonds", "formal_charge",
    "SMR_VSA3", "s_logP_vsa", "PEOE_VSA3", "PEOE_VSA6", "PEOE_VSA12"
]  # agregá todos los que correspondan


#Loop entrenamiento
for target in targets:
    target_df = data[data['target'] == target] #Divide cada target

    train_df = target_df[target_df['Data_split'] == 'train'] #Entrenamiento
    val_df = target_df[target_df['Data_split'] == 'validation'] #Validacion
    test_df = target_df[target_df['Data_split'] == 'test'] #Testeo

    X_train = train_df[descriptor_cols]  # Selecciona columnas de descriptores del set de entrenamiento
    y_train = train_df['bioactivity'] #Valores de bioactividad para predecir

    X_val = val_df[descriptor_cols]
    y_val = val_df['bioactivity']

    X_test = test_df[descriptor_cols]
    y_test = test_df['bioactivity']

    # Entrenar el modelo
    #Random Forest
    rf_model = RandomForestClassifier(oob_score=True, random_state=42)
    model= rf_model

    param_grid = {'max_depth': [2, 3, 4, 5, 6, 7,8,9],
    'n_estimators': [30, 50, 100, 200,300], #'objective': 'binary:logistic' fijarme como agregar esto pero qe no sea para str
    'min_samples_split': [2, 3, 4, 5]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='f1', cv=4)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    #VALIDACION
    # Realizar predicciones con RF en la validacion
    y_xgb_pred = best_model.predict(X_val)
    #metrica f1
    f1_xgb = f1_score(y_val, y_xgb_pred)
    #metrica roc auc
    roc_auc_xgb = roc_auc_score(y_val, y_xgb_pred)
    # Validar el modelo
    print(f'Target: {target} - Validation f1: {f1_xgb }')
    print(f'Target: {target} - Validation roc_auc_score: {roc_auc_xgb }')

    #TESTEO
   # Evaluar en el conjunto de prueba
    y_pred_test = best_model.predict(X_test)
    y_scores_test_1=best_model.predict_proba(X_test)[:,1]
    test_f1= f1_score(y_test, y_pred_test)
    test_roc_auc_xgb = roc_auc_score(y_test, y_scores_test_1)

    print(f'Target: {target} - Test Accuracy: {test_f1}')
    print(f'Target: {target} - Test roc_auc_score: {test_roc_auc_xgb }')

    #CAlCULO DE PROBABILIDADES PARA LAS CURVAS ROC testeo
    # Obtener las probabilidades predichas para ambas clases
    y_prob_test= best_model.predict_proba(X_test)

    # Crear un DataFrame con los resultados
    roc_data = pd.DataFrame({
        "comp_id": test_df['comp_id'],
        'smiles': test_df['smiles'],
        'true_label': y_test,
        'predicted_probability_class_0': y_prob_test[:, 0],  # Probabilidad de la clase 0
        'predicted_probability_class_1': y_prob_test[:, 1] }) #proba 1 (activos)
    # Guardar el DataFrame en un archivo CSV
    roc_data.to_csv(f'rocdata_RF/roc_data_RF_{target}.csv', index=False)
    print(f'ROC data for target {target} saved as roc_data_RF{target}.csv')

    #FEATURE IMPORTANCES DE CADA TARGET
    feature_importances = best_model.feature_importances_
    nombres_caracteristicas = [
    "numb", "C", "O", "N", "Cl", "F", "Br", "I", "nh2_pattern",
    "ring5_counts", "ring6_counts", "MolWeight", "Rings",
    "TPSA", "FpDM1", "RotBonds", "formal_charge",
    "SMR_VSA3", "s_logP_vsa", "PEOE_VSA3", "PEOE_VSA6", "PEOE_VSA12"]

    # PLOT FEATURE IMPORTANCES DE CADA TARGET
    plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico si es necesario
    plt.bar(range(len(feature_importances)), feature_importances, color="navy")
    plt.xticks(range(len(feature_importances)), nombres_caracteristicas, rotation=90)
    plt.xlabel("Características")
    plt.ylabel("Importancia")
    plt.title(f'Feature Importances for {target}')
    plt.tight_layout()
    # Guardar el gráfico de feature importance
    plt.savefig(f'feature_importances_plots_RF/{target}_feature_importances.pdf')
    plt.close()

    #RESULTADOS DE METRICAS DEL MODELO PARA CADA TARGET
    compuestos_totales = target_df['comp_id'].count() #CONTAR LA CANTIDAD DE COMPUESTOS TOTALES
    activos = target_df[target_df['bioactivity'] == 1].shape[0] #CONTAR LA CANTIDAD DE ACTIVOS DEL TARGET
    inactivos = target_df[target_df['bioactivity'] == 0].shape[0] #CONTAR LA CANTIDAD DE INACTIVOS DEL TARGET
    fila_actual = pd.DataFrame({
        'Target': [target],
        'Compuestos Totales': [compuestos_totales],
        'Activos': [activos],
        'Inactivos': [inactivos],
        'Validation F1': [f1_xgb],
        'Validation ROC AUC': [roc_auc_xgb],
        'Test F1': [test_f1],
        'Test ROC AUC': [test_roc_auc_xgb]
    })

    # Concatenar la fila actual al DataFrame de resultados
    resultados_df = pd.concat([resultados_df, fila_actual], ignore_index=True)
    # Guardar CSV de resultados → carpeta "resultados_RF"
    resultados_df.to_csv('resultados_RF/resultados_modelos_RF.csv', index=False)

    # GUARDAR MODELO → carpeta "modelos_RF"
    model_filename = f'modelos_RF/model_RF_{target}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)

    print(f'Model for target {target} saved as {model_filename}')