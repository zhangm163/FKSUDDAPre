# FKSUDDAPre：基于F检验与AMDKSU重采样技术的药物-疾病关联预测研究及可解释性分析

## 项目简介
FKSUDDAPre模型通过分子结构与疾病网络多模态特征融合，结合集成学习方法，实现药物-疾病关联的高效预测。模型包含特征提取、数据集平衡、特征选择和集成预测四大核心模块，支持从原始数据到最终预测结果的全流程自动化。

---

## 功能模块
1. **特征提取模块**  
   - 药物分子：Mol2vec对SMILES序列进行无监督表示，生成300维分子嵌入向量。
   - 疾病网络：DeepWalk在MeSH关系DAG上生成64维拓扑嵌入向量。

2. **平衡数据集构建模块**  
   - 改进KSU策略，结合K-means聚类，动态选择距离度量，缓解类别不平衡。

3. **关键特征选择模块**  
   - F检验等方法筛选判别力最强的特征。

4. **集成学习预测模块**  
   - 集成XGBoost、决策树、随机森林等多种基学习器。

---

## 数据输入与输出
- **输入**  
  - 药物：SMILES表达式（如`DrugInformation.csv`）
  - 疾病：MeSH术语DAG结构（如`MeSHFeatureGeneratedByDeepWalk.csv`）
  - 药物-疾病配对及标签（如`DrugDiseaseAssociationNumber.csv`）

- **输出**  
  - 药物-疾病关联预测概率或二分类结果（如`prediction_results.csv`）

---

## 环境依赖
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


## 目录结构
```
FKSUDDAPre/
├── extract/                # 特征提取相关脚本
├── dataprocess/            # 数据拼接与预处理
├── undersample/            # 数据集平衡方法
├── dimension_reduction/    # 特征选择与降维
├── model/                  # 各类模型实现
├── data/                   # 数据存放目录
├── train_DDA.py            # 主训练与预测入口
└── README.md
```

---

## 使用方法

### 1. 特征提取
#### 1.1 药物分子特征（Mol2vec）
```bash
python extract/mol2vec.py --input ./data/B-datasets/DrugInformation.csv --output ./data/B-datasets/feature_extraction/Drug_mol2vec.csv
```

#### 1.2 疾病网络特征（DeepWalk）
```bash
python extract/features.py --input ./data/B-datasets/MeSHFeatureGeneratedByDeepWalk.csv --output ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv
```

---

### 2. 特征拼接与样本构建
```bash
python dataprocess/association.py --drug ./data/B-datasets/feature_extraction/Drug_mol2vec.csv --disease ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv --pairs ./data/B-datasets/DrugDiseaseAssociationNumber.csv --output ./data/original_samples/association.csv
```
```bash
python dataprocess/disassociation.py --drug ./data/B-datasets/feature_extraction/Drug_mol2vec.csv --disease ./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv --pairs ./data/B-datasets/DrugDiseasedisAssociationNumber.csv --output ./data/original_samples/disassociation.csv
```
---

### 3. 数据集平衡（KSU等方法）
以汉明距离为例：
```bash
python undersample/KSU_Hamming.py --input ./data/original_samples/disassociation.csv --output ./data/undersample/disAssociaton/diaKSU_Hamming.csv
```
```bash
python ./data/undersample/merge.py 
```
---

### 4. 特征选择/降维
以F检验为例：
```bash
python dimension_reduction/f_classif.py --input ./data/after_dimension_reduction/KSU_Hamming.csv --output ./data/after_dimension_reduction/140/f_classif_KSU_Hamming140.csv
```

---
### 5. 模型训练与预测
直接运行主程序：
```bash
python train_DDA.py
```
- 默认读取`data/after_dimension_reduction/`下的特征文件
- 结果输出到`data/results/prediction_results.csv`

