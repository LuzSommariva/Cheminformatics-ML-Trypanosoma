# Cheminformatics-ML-Trypanosoma
Modelos de ML (Random Forest y XGBoost) para predecir la bioactividad binaria de compuestos contra proteínas de Trypanosoma cruzi, usando descriptores fisicoquímicos generados a partir de SMILES.
Se implementan dos enfoques de modelado: Random Forest (RF) y XGBoost, utilizando estos descriptores fisicoquímicos.

Para el cálculo de los descriptores se empleó RDKit, mientras que los modelos fueron entrenados utilizando scikit-learn y XGBoost.
Además, se utilizó DeepChem para realizar la partición del conjunto de datos mediante el algoritmo Scaffold Split, con el objetivo de garantizar una división estructuralmente diversa entre los conjuntos de entrenamiento, validación y prueba.
