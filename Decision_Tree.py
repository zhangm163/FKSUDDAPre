import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score,
                             recall_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, auc)
import os
from collections import Counter
datasets = [
    {
        "name": "KSU",
        "path": r"./data/undersample/KSU_Euclidean.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_KSU_Euclidean.csv"
    },
    {
        "name": "NeighbourhoodCleaningRule",
        "path": r"./data/undersample/NeighbourhoodCleaningRule.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_NeighbourhoodCleaningRule.csv"
    },
    {
        "name": "NearMiss",
        "path": r"./data/undersample/NearMiss.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_NearMiss.csv"
    },
    {
        "name": "OneSidedSelection",
        "path": r"./data/undersample/OneSidedSelection.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_OneSidedSelection.csv"
    },
    {
        "name": "RandomUnderSampler",
        "path": r"./data/undersample/RandomUnderSampler.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_RandomUnderSampler.csv"
    },
    {
        "name": "ClusterCentroids",
        "path": r"./data/undersample/ClusterCentroids.csv",
        "output_dir": r"./data/results/DT/",
        "results_file": r"./data/results/DT/DT_ClusterCentroids.csv"
    }
    #add more
]

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for dataset in datasets:
    print(f"Processing dataset: {dataset['name']}")
    df = pd.read_csv(dataset['path'])
    df.columns = df.columns.str.strip()
    drug_columns = [col for col in df.columns if col.endswith('_x')]
    disease_columns = [col for col in df.columns if col.endswith('_y')]
    df['drug'] = df['drug']
    df['disease'] = df['disease']
    df['label'] = df['label']

    X_drugs = df[drug_columns].values if drug_columns else np.zeros((df.shape[0], 0))
    X_diseases = df[disease_columns].values if disease_columns else np.zeros((df.shape[0], 0))
    X = np.hstack([X_drugs, X_diseases])
    y = df['label'].values
    print("Data distribution:")
    print(Counter(y))
    model = LinearDiscriminantAnalysis()
    results = []
    os.makedirs(dataset['output_dir'], exist_ok=True)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5,
                                   scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        sn = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision_val = precision_score(y_test, y_pred)
        recall_val = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sp = tn / (tn + fp)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        aupr = auc(recall, precision)
        results.append({
            "Fold": fold,
            "Accuracy (ACC)": acc,
            "AUC": auc_score,
            "AUPR": aupr,
            "Precision": precision_val,
            "Recall": recall_val,
            "F1 Score": f1,
            "MCC": mcc,
            "Sensitivity (Sn)": sn,
            "Specificity (Sp)": sp
        })
        print(f"Fold {fold}:")
        print(f"  Accuracy (ACC): {acc:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  AUPR: {aupr:.4f}")
        print(f"  Precision: {precision_val:.4f}")
        print(f"  Recall: {recall_val:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Sensitivity (Sn): {sn:.4f}")
        print(f"  Specificity (Sp): {sp:.4f}")
        print("-" * 50)
    results_df = pd.DataFrame(results)
    average_results = {
        "Fold": "Average",
        "Accuracy (ACC)": np.mean(results_df["Accuracy (ACC)"]),
        "AUC": np.mean(results_df["AUC"]),
        "AUPR": np.mean(results_df["AUPR"]),
        "Precision": np.mean(results_df["Precision"]),
        "Recall": np.mean(results_df["Recall"]),
        "F1 Score": np.mean(results_df["F1 Score"]),
        "MCC": np.mean(results_df["MCC"]),
        "Sensitivity (Sn)": np.mean(results_df["Sensitivity (Sn)"]),
        "Specificity (Sp)": np.mean(results_df["Specificity (Sp)"])
    }
    results_df = pd.concat([results_df, pd.DataFrame([average_results])], ignore_index=True)
    results_df.to_csv(dataset['results_file'], index=False)
    print(f"\nResults have been saved to {dataset['results_file']}")
