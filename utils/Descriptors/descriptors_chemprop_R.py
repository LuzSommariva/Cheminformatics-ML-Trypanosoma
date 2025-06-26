import os
import sys
import subprocess
import pandas as pd
import glob

# Verificar argumento
if len(sys.argv) < 2:
    print("Uso: python chemprop_fps.py <archivo_CSV>")
    sys.exit(1)

# Argumento recibido
input_filename = sys.argv[1]

# Rutas
input_dir = os.path.expanduser("~/Descargas/Targets")
output_dir = os.path.expanduser("~/Descargas/Targets/descrptores_chemprop_R")
model_path = "tests/data/example_model_v2_regression_mol.ckpt"

# Construir path completo de entrada
input_path = os.path.join(input_dir, input_filename)
base_name = os.path.splitext(input_filename)[0].upper()  # ej. "CHEMBL262"

# Verificar existencia del archivo de entrada
if not os.path.exists(input_path):
    print(f"❌ El archivo {input_path} no existe.")
    sys.exit(1)

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Ruta completa del output con .csv
output_base = os.path.join(output_dir, f"chemprop_fps_{base_name}.csv")

# Ejecutar chemprop fingerprint
print(f"🔄 Ejecutando chemprop fingerprint sobre {input_filename}...")
try:
    result = subprocess.run([
        "chemprop", "fingerprint",
        "--test-path", input_path,
        "--model-path", model_path,
        "--output", output_base,
        "--smiles-columns", "smiles",
        "--ffn-block-index", "-1"
    ], capture_output=True, text=True, check=True)
    print("✅ Chemprop ejecutado correctamente.")
except subprocess.CalledProcessError as e:
    print("❌ Error al ejecutar chemprop:")
    print(e.stderr)
    sys.exit(1)

# Buscar archivo generado
pattern = output_base.replace(".csv", "_*.csv")
descriptor_files = glob.glob(pattern)

if not descriptor_files:
    print(f"❌ No se generó ningún archivo de descriptores que coincida con: {pattern}")
    sys.exit(1)

descriptor_path = descriptor_files[0]  # Tomamos el primero, usualmente _0.csv
print(f"📄 Archivo de descriptores encontrado: {descriptor_path}")

# Leer ambos DataFrames
df_original = pd.read_csv(input_path)
df_descriptores = pd.read_csv(descriptor_path)

# Concatenar horizontalmente (asumiendo que están en el mismo orden)
df_merged = pd.concat([df_original.reset_index(drop=True), df_descriptores.reset_index(drop=True)], axis=1)

# Extraer target ID desde el nombre del archivo
target_id = base_name.replace("CHEMBL", "")
df_merged["target"] = target_id

# Guardar archivo final
final_output_path = os.path.join(output_dir, f"fps_merge_{base_name}.csv")
df_merged.to_csv(final_output_path, index=False)
print(f"✅ Archivo final guardado en: {final_output_path}")
