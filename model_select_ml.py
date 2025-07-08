import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:267].values.astype('float32')
    y = data['label'].values.astype('float32')
    print(f"Data shape: X: {X.shape}, y: {y.shape}")
    print(f"Sample data: X: {X[:5]}, y: {y[:5]}")
    return X, y

def evaluate_model(y_true, y_pred_prob, y_pred, threshold=0.5):
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    return acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val

def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def get_models():
    return {
        # 'XGBoost': lambda: XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42),
        'DecisionTree': lambda: DecisionTreeClassifier(random_state=42),
        # 'RandomForest': lambda: RandomForestClassifier(random_state=42),
        # 'LogisticRegression': lambda: LogisticRegression(max_iter=1000, solver='liblinear'),  # binary classification
        # 'SVM': lambda: SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
    }

def main(data_path, results_output_path, models_to_use, n_splits=10):
    X, y = load_data(data_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nProcessing Fold {fold}/{n_splits}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for model_name, model_class in models_to_use.items():
            print(f"Training model: {model_name}")
            model = model_class()
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_test, y_pred_prob)
            results.append({
                'Model': model_name,
                'Fold': fold,
                'Accuracy (ACC)': acc,
                'AUC': auc_score,
                'AUPR': aupr,
                'Precision': precision_val,
                'Recall': recall_val,
                'F1 Score': f1,
                'MCC': mcc,
                'Sensitivity (Sn)': sn,
                'Specificity (Sp)': sp
            })
            print(f"Fold {fold}, Model: {model_name}, AUC: {auc_score:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")
    save_results(results, results_output_path)


if __name__ == '__main__':
    models_to_use = get_models()
    model_names = '_'.join(models_to_use.keys())
    results_folder = f'./data/results/after_dimension_reduction/{model_names}'
    os.makedirs(results_folder, exist_ok=True)
    base_path = "./data/results/after_dimension_reduction"
    data_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                data_paths.append(full_path)
                print(f"find data：{full_path}")
    for data_path in data_paths:
        file_name = os.path.basename(data_path).split(".")[0]
        dataset_folder = os.path.join(results_folder, file_name)
        os.makedirs(dataset_folder, exist_ok=True)
        results_output_path = os.path.join(dataset_folder, f'{file_name}_results.csv')
        print(f"Starting processing: {data_path}")
        main(data_path, results_output_path, models_to_use)