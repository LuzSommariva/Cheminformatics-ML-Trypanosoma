



from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Fingerprints import FingerprintMols


# Importar libreria de RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors




import numpy as np
import pandas as pd


from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
import os
import sys

# Verificamos si se pasó el nombre del archivo
if len(sys.argv) < 2:
    print("Uso: python prueba.py <nombre_archivo.csv>")
    sys.exit(1)

# Obtener el nombre del archivo desde los argumentos
filename = sys.argv[1]

# Cargar el archivo CSV
total262 = pd.read_csv(filename)

# Extraer el ID del target desde el nombre del archivo (ej: "262" de "chembl262.csv")
# Convertimos a mayúsculas por si viene en minúsculas
target_id = os.path.splitext(os.path.basename(filename))[0].upper().replace("CHEMBL", "")

# Agregar la columna 'target'
total262["target"] = target_id

# Mostrar las primeras filas para confirmar
print(total262.head())



#CARGAR CADA TARGET QUE SE VAYA A ENTRENAR EN EL LOOP
#total262=pd.read_csv("CHEMBL262.csv")
#AGREGAR UNA COLUMNA DEL NOMBRE O ID DEL TARGET
#total262["target"]= "262"

# filtrar SMILES que no den error
SMILES = []
for i in range(len(total262['smiles'])):
  try:
    cs = Chem.CanonSmiles(total262["smiles"].iloc[i])
    SMILES.append(cs)
  except:
    print('Invalid SMILES:',total262['smiles'].iloc[i],i)

    # molecules from smiles
ms = [Chem.MolFromSmiles(smile) for smile in SMILES]
# number of atoms per molecule
numb = [mol.GetNumAtoms() for mol in ms]
# different atoms:
C = [] # carbon
O = [] # oxigen
N = [] # nitrogen
Cl=[]
F=[]
Br=[]
I=[]

nh2_pattern = Chem.MolFromSmarts("[NH2]")
nh2_counts = [len(mol.GetSubstructMatches(nh2_pattern)) for mol in ms]

ring5_counts = []
ring6_counts=[]


for mol in ms:
  c = 0
  o = 0
  n = 0
  cl=0
  f=0
  br=0
  i=0
  nh2_count = 0
  ring_info = mol.GetRingInfo()# Obtener sistemas de anillos saturados
  ring5_count = sum(1 for ring in ring_info.AtomRings() if len(ring) == 5)  # Contar anillos de 5 carbonos
  ring5_counts.append(ring5_count)
  ring6= sum(1 for ring in ring_info.AtomRings()if len(ring) == 6)  # Contar anillos de 5 carbonos
  ring6_counts.append(ring6)
  for atom in mol.GetAtoms():
    if atom.GetAtomicNum() == 6:
      c += 1
    if atom.GetAtomicNum() == 7:
      n += 1
    if atom.GetAtomicNum() == 8:
      o += 1
    if atom.GetAtomicNum() == 9:
      f += 1
    if atom.GetAtomicNum() == 17:
      cl += 1
    if atom.GetAtomicNum() == 35:
      br += 1
    if atom.GetAtomicNum() == 53:
      i += 1

  C.append(c)
  O.append(o)
  N.append(n)
  Cl.append(cl)
  F.append(f)
  Br.append(br)
  I.append(i)

# molecular weight
MolWeight = [Descriptors.ExactMolWt(mol) for mol in ms]
# rings
Rings = [Descriptors.RingCount(mol) for mol in ms]
# Total polar surface area
TPSA = [Descriptors.TPSA(mol) for mol in ms]
# Density Morgan FP
FpDM1 = [Descriptors.FpDensityMorgan1(mol) for mol in ms]  #aca me esta tirando error
# rotatable bonds
RotBonds = [Descriptors.NumRotatableBonds(mol) for mol in ms]
# Obtener la carga formal de la molécula
formal_charge =[Chem.rdmolops.GetFormalCharge(mol) for mol in ms]
# Calcular el logP (coeficiente de partición octanol-agua) de la molécula
#logp =[Descriptors.MolLogP(mol) for mol in ms]

#DESCRITORES VSA
SMR_VSA3= [Descriptors.SMR_VSA3(mol) for mol in ms] #SMR :calcula el aporte de cada atomo a la refractividad molar, la polarizabilidad.
s_logP_vsa= [Crippen.MolLogP(mol)for mol in ms] #logaritmo de partiicion
PEOE_VSA3 = [Descriptors.PEOE_VSA3(mol) for mol in ms] #PEOE DISTRIBUCION DE CARGA PARCIAL DE LA MOLECULA : áreas de la superficie de la molécula con cargas parciales negativas más significativas. (1-5)
PEOE_VSA6= [Descriptors.PEOE_VSA6(mol) for mol in ms] #áreas neutras
PEOE_VSA12 =[Descriptors.PEOE_VSA12(mol) for mol in ms] #Capturan las áreas con cargas parciales positivas

#DATAFRAME DE LOS DESCRIPTORES
descriptors = pd.DataFrame(list(zip(numb, C, O, N,F,Br,I,Cl, nh2_counts,ring5_counts,ring6_counts,MolWeight, Rings, TPSA, FpDM1, RotBonds,formal_charge,SMR_VSA3,s_logP_vsa,PEOE_VSA3,PEOE_VSA6,PEOE_VSA12)),index=total262.index)

# Lista de nombres de características
nombres_caracteristicas = [
    "numb", "C", "O", "N", "Cl", "F", "Br", "I", "nh2_pattern",
    "ring5_counts", "ring6_counts", "MolWeight", "Rings",
    "TPSA", "FpDM1", "RotBonds", "formal_charge",
    "SMR_VSA3", "s_logP_vsa", "PEOE_VSA3", "PEOE_VSA6", "PEOE_VSA12"
]

# Crear un diccionario de mapeo de índices a nombres de características
column_mapping = {i: nombres_caracteristicas[i] for i in range(len(nombres_caracteristicas))}

# Renombrar las columnas de X_train, X_val, y X_test
descriptors.rename(columns=column_mapping, inplace=True)

target_df = total262.reset_index(drop=True)
descriptors_df = descriptors.reset_index(drop=True)

# Concatenar los dataframes
todo= pd.concat([target_df, descriptors_df], axis=1)

from sklearn.preprocessing import MinMaxScaler

# Definir las columnas a normalizar
columnas_a_normalizar =  [
    "numb", "C", "O", "N", "Cl", "F", "Br", "I", "nh2_pattern",
    "ring5_counts", "ring6_counts", "MolWeight", "Rings",
    "TPSA", "FpDM1", "RotBonds", "formal_charge",
    "SMR_VSA3", "s_logP_vsa", "PEOE_VSA3", "PEOE_VSA6", "PEOE_VSA12"
]

# Crear el scaler
scaler = MinMaxScaler()

# Ajustar y transformar solo esas columnas
todo[columnas_a_normalizar] = scaler.fit_transform(todo[columnas_a_normalizar])

# Obtener el nombre base del archivo sin extensión
base_filename = os.path.splitext(os.path.basename(filename))[0]

# Guardar el archivo con el nombre "descriptores_<nombre_original>.csv"
# Crear carpeta para guardar descriptores
output_dir = "descriptores"
os.makedirs(output_dir, exist_ok=True)

# Guardar el archivo con nombre único basado en el target
output_path = os.path.join(output_dir, f"descriptores_{base_filename}.csv")
todo.to_csv(output_path, index=False)

print(f"Archivo guardado en: {output_path}")

#print(f"Archivo guardado como: {output_filename}")