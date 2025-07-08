import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")
sys.path.append(model_dir)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from decision_tree_model import DecisionTreeModel
from svm_model import LinearSVCModel
from DNN import DNN
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from logistic_model import LogisticRegressionModel
from TextRCNN import TextRCNN
from scipy.stats import mode


def load_data(file_path, has_label=True):
    data = pd.read_csv(file_path)
    drug_ids = data.iloc[:, 0].values
    disease_ids = data.iloc[:, 1].values
    if has_label:
        X = data.iloc[:, 3:].values.astype('float32')
        y = data['label'].values.astype('float32')
        y = torch.tensor(y)
    else:
        X = data.iloc[:, 2:].values.astype('float32')
        y = None
    X = torch.tensor(X)
    print(f"Data shape: X: {X.shape}, y: {y.shape if y is not None else 'No label'}")
    print(f"Sample data: X: {X[:5]}, y: {y[:5] if y is not None else 'No label'}")
    return drug_ids, disease_ids, TensorDataset(X, y) if y is not None else TensorDataset(X)

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
            y_pred_logits = model(X_batch).squeeze()
            loss = criterion(y_pred_logits, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

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
        print(f"Epoch {epoch + 1}/{epochs}, Validation AUC: {val_auc:.4f}")

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
    y_pred_prob = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_logits = model(X_batch).squeeze()
            y_pred = sigmoid(y_pred_logits).cpu().numpy()
            y_pred_prob.extend(y_pred)
    return y_pred_prob


def get_models_and_paths():
    return {
        'RF': {
            'model_class': RandomForestModel,
            'train_path': r'/home/zhangcy/file/FKSUDDAPre/data/after_dimension_reduction/90/f_classif_KSU_Hamming90.csv'
        },
        'DecisionTree': {
            'model_class': DecisionTreeModel,
            'train_path': r'/home/zhangcy/file/FKSUDDAPre/data/after_dimension_reduction/140/f_classif_KSU_Hamming140.csv'
        },
        'XGBoost': {
            'model_class': XGBoostModel,
            'train_path': r'/home/zhangcy/file/FKSUDDAPre/data/after_dimension_reduction/160/f_classif_KSU_Hamming160.csv'
        },
        # add more
    }

def initialize_model(model_class, input_size, output_size):
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

def save_hard_voting_results(results, output_path):
    hard_voting_results = [result for result in results if result['Model'] == 'HardVoting']
    hard_voting_df = pd.DataFrame(hard_voting_results)
    hard_voting_df.to_csv(output_path, index=False)
    print(f"Hard voting results saved to {output_path}")


def align_features(X_train, X_test, train_feature_dim, test_feature_dim):
    if train_feature_dim == test_feature_dim:
        return X_train, X_test
    print(f"Aligning features from {train_feature_dim}D to {test_feature_dim}D")
    if train_feature_dim < test_feature_dim:
        padding = torch.zeros((X_train.shape[0], test_feature_dim - train_feature_dim))
        X_train = torch.cat([X_train, padding], dim=1)
    else:
        X_train = X_train[:, :test_feature_dim]
    return X_train, X_test


def main(test_data_path, results_output_path, epochs=20, batch_size=64,
         learning_rate=0.001, n_splits=10, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_drug_ids, test_disease_ids, test_dataset = load_data(test_data_path, has_label=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_feature_dim = test_dataset.tensors[0].shape[1]
    results = []
    all_model_predictions = []
    models_and_paths = get_models_and_paths()

    for model_name, model_info in models_and_paths.items():
        print(f"\n{'=' * 50}\nProcessing model: {model_name}")
        train_path = model_info['train_path']
        print(f"Using training data: {train_path}")
        train_drug_ids, train_disease_ids, train_dataset = load_data(train_path, has_label=True)
        train_feature_dim = train_dataset.tensors[0].shape[1]

        model_class = model_info['model_class']
        model = initialize_model(model_class, input_size=train_feature_dim, output_size=1)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"\nFold {fold + 1}/{n_splits}")

            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            X_train = train_subset.dataset.tensors[0][train_idx]
            X_val = val_subset.dataset.tensors[0][val_idx]
            X_train, X_val = align_features(X_train, X_val, train_feature_dim, train_feature_dim)
            train_subset = TensorDataset(X_train, train_subset.dataset.tensors[1][train_idx])
            val_subset = TensorDataset(X_val, val_subset.dataset.tensors[1][val_idx])
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            if isinstance(model, nn.Module):
                model = model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
                train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs,
                            early_stopping_patience)
                y_pred_prob = test_model(model, val_loader, device)
                y_pred = (y_pred_prob >= 0.5).astype(int)
                y_true = val_subset.dataset.tensors[1][val_idx].numpy()
            else:
                X_train_np = train_subset.tensors[0].numpy()
                y_train_np = train_subset.tensors[1].numpy()
                X_val_np = val_subset.tensors[0].numpy()
                y_true = val_subset.tensors[1].numpy()
                if X_train_np.shape[1] != X_val_np.shape[1]:
                    min_dim = min(X_train_np.shape[1], X_val_np.shape[1])
                    X_train_np = X_train_np[:, :min_dim]
                    X_val_np = X_val_np[:, :min_dim]
                y_pred, y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_val_np)
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
                y_true, y_pred_prob, y_pred)
            fold_result = {
                'Model': model_name,
                'Fold': fold + 1,
                'TrainingData': os.path.basename(train_path),
                'FeatureDim': train_feature_dim,
                'Accuracy': acc,
                'AUC': auc_score,
                'AUPR': aupr,
                'F1 Score': f1,
                'MCC': mcc,
                'Sensitivity': sn,
                'Specificity': sp,
                'Precision': precision_val,
                'Recall': recall_val,
                'Mean Probability': np.mean(y_pred_prob),
                'Median Probability': np.median(y_pred_prob),
                'Min Probability': np.min(y_pred_prob),
                'Max Probability': np.max(y_pred_prob)
            }
            fold_results.append(fold_result)
            print(f"Fold {fold + 1} Results:")
            print(f"Accuracy: {acc:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr:.4f}")
            print(f"F1: {f1:.4f}, MCC: {mcc:.4f}, Sensitivity: {sn:.4f}, Specificity: {sp:.4f}")
        results.extend(fold_results)
        print("\nMaking predictions on test set...")
        if isinstance(model, nn.Module):
            original_test_loader = test_loader
            if train_feature_dim != test_feature_dim:
                X_test = test_dataset.tensors[0]
                X_test_aligned = X_test[:, :min(train_feature_dim, test_feature_dim)]
                test_dataset_aligned = TensorDataset(X_test_aligned)
                test_loader = DataLoader(test_dataset_aligned, batch_size=batch_size, shuffle=False)
            y_pred_prob = test_model(model, test_loader, device)
            test_loader = original_test_loader
        else:
            X_train_np = train_dataset.tensors[0].numpy()
            y_train_np = train_dataset.tensors[1].numpy()
            X_test_np = test_dataset.tensors[0].numpy()
            if X_train_np.shape[1] != X_test_np.shape[1]:
                min_dim = min(X_train_np.shape[1], X_test_np.shape[1])
                X_train_np = X_train_np[:, :min_dim]
                X_test_np = X_test_np[:, :min_dim]
            y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_test_np)[1]
        all_model_predictions.append({
            'name': model_name,
            'prob': y_pred_prob,
            'feature_dim': train_feature_dim
        })
        test_scores = pd.DataFrame({
            'Drug': test_drug_ids,
            'Disease': test_disease_ids,
            'Score': y_pred_prob,
            'Model': model_name,
            'TrainingData': os.path.basename(train_path),
            'FeatureDim': train_feature_dim
        })
        test_scores_output_path = results_output_path.replace('.csv', f'_{model_name}_test_scores.csv')
        test_scores.to_csv(test_scores_output_path, index=False)
        print(f"Test predictions saved to {test_scores_output_path}")
    if len(all_model_predictions) > 1:
        print("\nPerforming model ensemble with hard voting...")
        min_length = min(len(p['prob']) for p in all_model_predictions)
        aligned_predictions = [p['prob'][:min_length] for p in all_model_predictions]
        hard_votes, _ = mode(np.stack([(pred >= 0.5).astype(int) for pred in aligned_predictions]), axis=0)
        hard_votes = hard_votes.flatten()
        final_scores = []
        for i in range(min_length):
            supported_models = [pred[i] for pred, model in zip(aligned_predictions, all_model_predictions)
                                if (pred[i] >= 0.5) == (hard_votes[i] == 1)]
            final_scores.append(np.mean(supported_models) if supported_models else 0.5)
        final_test_scores = pd.DataFrame({
            'Drug': test_drug_ids[:min_length],
            'Disease': test_disease_ids[:min_length],
            'Score': final_scores,
            'Vote': hard_votes,
            'Model': 'Ensemble(HardVoting)'
        })
        final_test_scores_output_path = results_output_path.replace('.csv', '_ensemble_test_scores.csv')
        final_test_scores.to_csv(final_test_scores_output_path, index=False)
        print(f"Ensemble predictions saved to {final_test_scores_output_path}")
        ensemble_result = {
            'Model': 'HardVoting',
            'Fold': 'Ensemble',
            'TrainingData': 'Multiple',
            'FeatureDim': 'Various',
            'Accuracy': 'N/A',
            'AUC': 'N/A',
            'AUPR': 'N/A',
            'F1 Score': 'N/A',
            'MCC': 'N/A',
            'Sensitivity': 'N/A',
            'Specificity': 'N/A',
            'Precision': 'N/A',
            'Recall': 'N/A',
            'Mean Probability': np.mean(final_scores),
            'Median Probability': np.median(final_scores),
            'Min Probability': np.min(final_scores),
            'Max Probability': np.max(final_scores)
        }
        results.append(ensemble_result)
    print("\nSaving all evaluation results...")
    save_results(results, results_output_path)
    hard_voting_output_path = results_output_path.replace('.csv', '_hard_voting_results.csv')
    save_hard_voting_results(results, hard_voting_output_path)
    print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    test_data_path = './data/candidate_drug/candidate_drug_top3_disease567.csvv'
    results_folder = './data/results/candidate_drug'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_output_path = os.path.join(results_folder, '567chills_multi_model.csv')
    main(test_data_path=test_data_path,
         results_output_path=results_output_path,
         epochs=20,
         batch_size=64,
         learning_rate=0.001,
         n_splits=10,
         early_stopping_patience=5)
