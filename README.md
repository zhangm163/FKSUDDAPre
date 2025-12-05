# FKSUDDAPre: A Drug–Disease Association Prediction Framework Based on F-TEST Feature Selection and AMDKSU Resampling with Interpretability Analysis

## Project Overview
The FKSUDDAPre model enables efficient prediction of drug–disease associations by integrating multimodal features from drug molecular structures and disease networks, combined with ensemble learning strategies. The framework comprises four core modules: feature extraction, dataset balancing, feature selection, and ensemble prediction, supporting an end-to-end pipeline from raw data to final prediction results.

## Requirements:
- Python >= 3.7
- numpy
- pandas
- scikit-learn
- xgboost
- joblib
- torch
- rdkit-pypi
- gensim
- networkx
- matplotlib
- seaborn
- scipy
- lime
- PyQt5
- hyperfast


## Directory structure
```
FKSUDDAPre/
├── extract/                # Feature extraction related scripts
├── dataprocess/            # Data concatenation and preprocessing
├── undersample/            # Data set balancing 
├── dimension_reduction/    # Feature selection and dimensionality reduction
├── model/                  # Implementation of various models
├── data/                   # datasets
├── train_DDA.py            # Main training and prediction entry point
└── README.md
```

---

## How to use

### 1. Feature extraction
#### 1.1 Drug molecule features (Mol2vec_k-bert)
```bash
python extract/mol2vec.py --input ./data/B-datasets/DrugInformation.csv --output ./data/B-datasets/feature_extraction/Drug_mol2vec.csv
python extract/k-bert.py
python data/B-datasets/feature_extraction/mol2vec_kbert.py
```

#### 1.2 disease networks features（DeepWalk）
```bash
python extract/features.py --input ./data/B-datasets/MeSHFeatureGeneratedByDeepWalk.csv --output ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv
```

---

### 2. Feature Concatenation and Sample Construction
```bash
python dataprocess/association.py --drug ./data/B-datasets/feature_extraction/DrugFeature.csv --disease ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv --pairs ./data/B-datasets/DrugDiseaseAssociationNumber.csv --output ./data/original_samples/association.csv
```
```bash
python dataprocess/disassociation.py --drug ./data/B-datasets/feature_extraction/Drug_mol2vec.csv --disease ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv --pairs ./data/B-datasets/DrugDiseasedisAssociationNumber.csv --output ./data/original_samples/disassociation.csv
```
---

### 3. Balanced Dataset Construction
Using Hamming Distance as an example:
```bash
python undersample/KSU_Hamming.py --input ./data/original_samples/disassociation.csv --output ./data/undersample/disAssociaton/diaKSU_Hamming.csv
```
```bash
python ./data/undersample/merge.py 
```
---

### 4. Feature Selection / Dimensionality Reduction
Using F-test as an example:
```bash
python dimension_reduction/f_classif.py --input ./data/after_dimension_reduction/KSU_Hamming.csv --output ./data/after_dimension_reduction/140/f_classif_KSU_Hamming140.csv
```

---
### 5. Model Training and Prediction
Run the main script directly:
```bash
python train_DDA.py
```
- Run the main script directly:`data/after_dimension_reduction/`
- Prediction results are saved to`data/results/prediction_results.csv`

---
## About Predictor
### Download and Setup
- The predictor can be downloaded at  https://pan.baidu.com/s/15Ifynpi2r_ABVGUNdaYMug?pwd=tg9x 
Access code: tg9x  
Download and get a zip package，unzip it ，find the predictor.exe and click it to run the predictor.  
To facilitate online prediction, we have developed an online predictor based on Python.
- The pre-trained files for drug feature extraction can be downloaded from https://pan.baidu.com/s/11f68mQ96-sRCyw55WpWV8g 
Access code: nhtq
### How to Use Predictor
First, click the "Select Model" button at the top of the interface and choose the local model file. The file format should be "pkl".  
After selecting the model file, choose the prediction file. Click the "Select Data" button at the top of the interface and choose the local data file. The file format should be "csv".  
After selecting the test file, click "Start Prediction" and wait for the prediction results. The results will be displayed in a list format on the interface, as shown in Figure 5-6. The list contains 4 columns: the first column is the drug ID, the second is the disease ID, the third column is the predicted score, and the fourth column is the predicted label, represented as "negative" or "positive".  
There is a button labeled "Save Results" at the top of the interface. By clicking this button, users can easily save the model's prediction results in CSV file format to their local computer.  


