import os
import sys
from sklearn.metrics import auc as sklearn_auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")
sys.path.append(model_dir)
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from feature_selector import FeatureSelector
from decision_tree_model import DecisionTreeModel
from svm_model import LinearSVCModel
from DNN import DNN
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from logistic_model import LogisticRegressionModel
from TextRCNN import TextRCNN
from hyperfast_model import HyperFastModel
from sklearn.linear_model import LogisticRegression


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    return TensorDataset(X, y)


def evaluate_model(y_true, y_pred_prob, y_pred, threshold=0.5):
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = sklearn_auc(recall, precision)
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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20,
                early_stopping_patience=5):
    model.train()
    best_val_auc = 0.0
    early_stopping_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_logits = model(X_batch)
            y_pred_logits = y_pred_logits.squeeze()
            loss = criterion(y_pred_logits, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        y_true, y_pred_prob = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred_logits = model(X_batch).squeeze()
                y_pred = torch.sigmoid(y_pred_logits).cpu().numpy()
                y_true.extend(y_batch.cpu().numpy())
                y_pred_prob.extend(y_pred)
            val_auc = roc_auc_score(y_true, y_pred_prob)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        model.train()
        scheduler.step(epoch_loss)


def test_model(model, test_loader, device):
    sigmoid = nn.Sigmoid()
    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch).squeeze()
            y_pred = sigmoid(y_pred_logits).cpu().numpy()
            y_true.extend(y_batch.cpu().numpy())
            y_pred_prob.extend(y_pred)
    return y_true, y_pred_prob


def get_models():
    return {
        'DecisionTree': DecisionTreeModel,
        'XGBoost': XGBoostModel,
        'RandomForest': RandomForestModel,
        'HyperFast': HyperFastModel,
    }


def initialize_model(model_class, input_size, output_size):
    if model_class == DNN:
        model = model_class(input_size, output_size)
        model.apply(init_weights)
    elif model_class == TextRCNN:
        model = model_class(input_size, output_size)
    else:
        model = model_class(random_state=42)
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)


def save_ensemble_results(results, output_path):
    ensemble_results = [result for result in results if result['Model'] == 'Stacking']
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df.to_csv(output_path, index=False)
    print(f"Stacking results saved to {output_path}")


def save_ensemble_model(model_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model_dict, os.path.join(output_dir, 'ensemble_model.pkl'))


def main(data_paths, results_output_path, models_to_use, epochs=20, batch_size=64, learning_rate=0.001, n_splits=10,
         early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {model_name: load_data(data_path) for model_name, data_path in data_paths.items()}
    results = []
    fold = 1
    dataset_sizes = [len(dataset) for dataset in datasets.values()]
    if len(set(dataset_sizes)) != 1:
        raise ValueError(f"All datasets must have the same number of samples! Found sizes: {dataset_sizes}")

    first_model_name = list(models_to_use.keys())[0]
    X = datasets[first_model_name].tensors[0]

    trained_models = {model_name: [] for model_name in models_to_use.keys()}
    all_roc_data = {model_name: {'y_true': [], 'y_pred_prob': []} for model_name in models_to_use.keys()}
    all_roc_data['Stacking'] = {'y_true': [], 'y_pred_prob': []}

    # --- PHASE 1: Generating Out-of-Fold (OOF) predictions for base models ---
    print("--- Phase 1: Generating Out-of-Fold (OOF) predictions for base models ---")

    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        print(f"Processing fold {fold}/{n_splits}")
        y_true = None

        for model_name, dataset in datasets.items():
            print(f"Processing model: {model_name}")
            X_data = dataset.tensors[0]
            y_data = dataset.tensors[1]

            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model_class = models_to_use[model_name]
            model = initialize_model(model_class, input_size=X_data.shape[1], output_size=1)

            if isinstance(model, nn.Module):
                model = model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
                train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs,
                            early_stopping_patience)
                y_true, y_pred_prob = test_model(model, test_loader, device)
            else:
                X_train_np = X_train.numpy()
                y_train_np = y_train.numpy()
                X_test_np = X_test.numpy()
                y_test_np = y_test.numpy()
                y_pred, y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_test_np)
                y_true = y_test_np
            y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)

            trained_models[model_name].append(model)

            all_roc_data[model_name]['y_true'].extend(y_true)
            all_roc_data[model_name]['y_pred_prob'].extend(y_pred_prob)

            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, y_pred_prob,
                                                                                              y_pred)
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

        print(f"Fold {fold} completed.")
        print("-" * 100)
        fold += 1

    # --- PHASE 2: Train and Evaluate Stacking (Level 1) Model ---
    print("\n" + "--- Phase 2: Training and Evaluating Stacking (Level 1) Model ---")

    meta_features = []
    model_names_order = list(models_to_use.keys())
    print(f"Meta-features (Level 1 inputs) order: {model_names_order}")

    for model_name in model_names_order:
        meta_features.append(all_roc_data[model_name]['y_pred_prob'])

    X_meta = np.stack(meta_features, axis=1)
    y_meta = np.array(all_roc_data[first_model_name]['y_true'])

    meta_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=43)

    stacking_y_true_agg = []
    stacking_y_prob_agg = []

    fold = 1
    for meta_train_idx, meta_test_idx in meta_kfold.split(X_meta, y_meta):
        X_meta_train, X_meta_test = X_meta[meta_train_idx], X_meta[meta_test_idx]
        y_meta_train, y_meta_test = y_meta[meta_train_idx], y_meta[meta_test_idx]

        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(X_meta_train, y_meta_train)

        y_pred_prob_stacking = meta_model.predict_proba(X_meta_test)[:, 1]
        y_pred_stacking = meta_model.predict(X_meta_test)

        stacking_y_true_agg.extend(y_meta_test)
        stacking_y_prob_agg.extend(y_pred_prob_stacking)

        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_meta_test,
                                                                                          y_pred_prob_stacking,
                                                                                          y_pred_stacking)
        results.append({
            'Model': 'Stacking',
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
        print(f"Stacking Fold {fold}/{n_splits} - AUC: {auc_score:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
        fold += 1

    all_roc_data['Stacking']['y_true'] = stacking_y_true_agg
    all_roc_data['Stacking']['y_pred_prob'] = stacking_y_prob_agg

    save_results(results, results_output_path)

    ensemble_output_path = results_output_path.replace('.csv', '_stacking_results.csv')
    save_ensemble_results(results, ensemble_output_path)

    model_save_dir = os.path.join(os.path.dirname(results_output_path), 'saved_models')
    save_ensemble_model(trained_models, model_save_dir)


if __name__ == '__main__':
    if '__file__' not in globals():
        __file__ = 'your_script_name.py'

    data_paths = {
        'DecisionTree': './data/after_dimension_reduction/140/f_classif_KSU_Hamming140.csv',
        'RandomForest': './data/after_dimension_reduction/110/f_classif_KSU_Hamming110.csv',
        'XGBoost': './data/after_dimension_reduction/160/f_classif_KSU_Hamming160.csv',
        'HyperFast': './data/after_dimension_reduction/80/f_classif_KSU_Hamming80.csv',
    }

    results_folder = r'./data/results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_output_path = os.path.join(results_folder, 'prediction_results_hyperfast.csv')
    models_to_use = get_models()

    main(data_paths, results_output_path, models_to_use)