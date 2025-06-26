# Cheminformatics-ML-Trypanosoma
## Machine Learning Models for Binary Bioactivity Prediction Against *Trypanosoma cruzi* Proteins

This project aims to predict the binary bioactivity (active/inactive) of chemical compounds against selected *Trypanosoma cruzi* protein targets using physicochemical descriptors derived from SMILES strings.

Two machine learning algorithms are implemented:
- **Random Forest**: An ensemble method that builds multiple decision trees and aggregates their results.
- **XGBoost**: A gradient boosting algorithm known for its speed and performance on structured data.

Both models are trained using the **scikit-learn** library.

---

## Data Splitting

To ensure a diverse and unbiased partitioning of the data, the **Scaffold Split** algorithm from the [DeepChem](https://deepchem.io/) library is used. This method splits compounds based on molecular scaffolds, promoting generalization and avoiding overfitting.

---

## Targets

The `targets/` folder contains the CSV files used to train the models. Seven targets were selected based on:
- The number of compounds per target.
- The balance between active and inactive compounds.

---

## Descriptors

Physicochemical descriptors are required for model training. Six different types are available or can be generated:

### 1. **RDKit Descriptors**
- descriptors_rdkit: 21 physicochemical descriptors computed using RDKit.

### 2. **Chemprop Descriptors**
- `Descriptors_chemprop_C`: 300 descriptors from a Chemprop classification model.
- `Descriptors_chemprop_R`: 300 descriptors from a Chemprop regression model.
- To compute these descriptors, install the Chemprop repository:  
   [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)

### 3. **Fingerprint-Based Descriptors**
- `fingerprints_rdkit`: 200 descriptors representing floating-point physicochemical properties and topological counts.
- `fingerprints_MACCS`: 166 binary descriptors indicating the presence of specific substructures (e.g., rings, functional groups).
- `fingerprints_ECFP`: 1024-bit extended circular fingerprints capturing atomic environments using hashed subgraphs.

---

##  Models

After computing the descriptors, the following models can be trained:

- **Random Forest**  
  Builds multiple decision trees and aggregates their outputs to reduce variance and overfitting.

- **XGBoost**  
  A regularized and optimized gradient boosting method that builds trees sequentially to minimize prediction error.

---




