# Importar libreria de RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import numpy as np
import pandas as pd


from rdkit.Chem import Crippen

import os
import sys

from skfp.preprocessing import MolFromSmilesTransformer
from skfp.fingerprints import MACCSFingerprint

# Verificamos si se pasó el nombre del archivo
if len(sys.argv) < 2:
    print("Uso: python prueba.py <nombre_archivo.csv>")
    sys.exit(1)

# Obtener el nombre del archivo desde los argumentos
filename = sys.argv[1]

# Cargar el archivo CSV
data = pd.read_csv(filename)

# Extraer el ID del target desde el nombre del archivo (ej: "262" de "chembl262.csv")
# Convertimos a mayúsculas por si viene en minúsculas
target_id = os.path.splitext(os.path.basename(filename))[0].upper().replace("CHEMBL", "")

# Agregar la columna 'target'
data["target"] = target_id

# Mostrar las primeras filas para confirmar
print(data.head())

# filtrar SMILES que no den error
SMILES = []
for i in range(len(data['smiles'])):
  try:
    cs = Chem.CanonSmiles(data["smiles"].iloc[i])
    SMILES.append(cs)
  except:
    print('Invalid SMILES:',data['smiles'].iloc[i],i) 
    
#Modelo para calcular los fingerprints
mol_from_smiles = MolFromSmilesTransformer()
# Crear una lista de moléculas a partir de los SMILES válidos
mols = mol_from_smiles.transform(SMILES)


fp_maccs = MACCSFingerprint(n_jobs=-1)

X_maccs = fp_maccs.transform(mols)

print(f"MACCS Fingerprints shape: {X_maccs.shape}")
print(f"MACCS Fingerprints  example values: {X_maccs[0, :10]}")

target_df = data.reset_index(drop=True)

fingerprint_df = pd.DataFrame(X_maccs).reset_index(drop=True)

# Obtener el nombre base del archivo sin extensión
base_filename = os.path.splitext(os.path.basename(filename))[0]
# Concatenar los dataframes
todo= pd.concat([target_df, fingerprint_df], axis=1)

output_dir = "fingerprints_MACCS"
os.makedirs(output_dir, exist_ok=True)

# Guardar el archivo con nombre único basado en el target
output_path = os.path.join(output_dir, f"fingerprints_MACCS_{base_filename}.csv")
todo.to_csv(output_path, index=False)

print(f"Archivo guardado en: {output_path}")

#print(f"Archivo guardado como: {output_filename}")
