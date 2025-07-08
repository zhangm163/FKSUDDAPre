import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")
sys.path.append(model_dir)
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
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

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    # print(f"Data shape: X: {X.shape}, y: {y.shape}")
    # print(f"Sample data: X: {X[:5]}, y: {y[:5]}")
    return TensorDataset(X, y)


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
            y_pred_logits = model(X_batch) 
            y_pred_logits = y_pred_logits.squeeze() 
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
        'RandomForest': RandomForestModel
        # 'DNN': DNN, 
        # 'TextRCNN': TextRCNN,    
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


from scipy.stats import mode
def save_hard_voting_results(results, output_path):
    hard_voting_results = [result for result in results if result['Model'] == 'HardVoting']
    hard_voting_df = pd.DataFrame(hard_voting_results)
    hard_voting_df.to_csv(output_path, index=False)
    print(f"Hard voting results saved to {output_path}")

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
    X = datasets[list(datasets.keys())[0]].tensors[0]
    y = datasets[list(datasets.keys())[0]].tensors[1]
    trained_models = {model_name: [] for model_name in models_to_use.keys()}

    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        print(f"Processing fold {fold}/{n_splits}")
        all_model_predictions = []
        all_predictions_prob = []
        y_true = None
        for model_name, dataset in datasets.items():
            print(f"Processing model: {model_name}")
            X = dataset.tensors[0]
            y = dataset.tensors[1]

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model_class = models_to_use[model_name]
            model = initialize_model(model_class, input_size=X.shape[1], output_size=1)

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
            all_model_predictions.append(y_pred)
            all_predictions_prob.append(y_pred_prob)

            trained_models[model_name].append(model)

            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, y_pred_prob, y_pred)
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

        final_predictions, _ = mode(np.stack(all_model_predictions), axis=0)
        final_predictions = final_predictions.flatten()
        avg_prob = np.mean(np.stack(all_predictions_prob), axis=0)
        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true,avg_prob, final_predictions)
        results.append({
            'Model': 'HardVoting',
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
        print(f"Hard Voting Evaluation Results - ACC: {acc:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr:.4f}, "
              f"F1: {f1:.4f}, MCC: {mcc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}")
        print("-" * 100)
        fold += 1

    save_results(results, results_output_path)

    hard_voting_output_path = results_output_path.replace('.csv', '_hard_voting.csv')
    save_hard_voting_results(results, hard_voting_output_path)
    model_save_dir = os.path.join(os.path.dirname(results_output_path), 'saved_models')
    save_ensemble_model(trained_models, model_save_dir)
    
if __name__ == '__main__':

    data_paths = {
        'DecisionTree': './data/after_dimension_reduction/140/f_classif_KSU_Hamming140.csv',
        'RandomForest': './data/after_dimension_reduction/90/f_classif_KSU_Hamming90.csv',
        'XGBoost': './data/after_dimension_reduction/160/f_classif_KSU_Hamming160.csv',
        # 'DNN':          './data/after_dimension_reduction/110/f_classif_KSU_Manhattan110.csv',
        # 'TextRCNN':     './data/after_dimension_reduction/90/RFE_KSU_Chebyshev90.csv',
    }
    results_folder = r'./data/results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_output_path = os.path.join(results_folder, 'prediction_results.csv')
    models_to_use = get_models()
    main(data_paths, results_output_path, models_to_use)
