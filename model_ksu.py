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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from DNN import DNN
from decision_tree_model import DecisionTreeModel
from random_forest_model import RandomForestModel
from GRU import GRU
from RNN import RNN
from TextRCNN import TextRCNN
from xgboost_model import XGBoostModel


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

def get_models(model_name):
    model_map = {
        #'BiLSTM': BiLSTM,
        # 'GRU': GRU,
        # 'DNN': DNN,
        # 'RNN': RNN,
          'TextRCNN': TextRCNN,
        # 'MLP': model_MLP,
        # 'AttentionLSTM': AttentionLSTM
        # 'DNNAttention': DNNAttention
    }
    return model_map[model_name]

def initialize_model(model_class, input_size, output_size):
    model = model_class(input_size, output_size)
    model.apply(init_weights)
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

def main(data_path, results_base_path, model_name, epochs=20, batch_size=64, learning_rate=0.001, n_splits=10, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_data(data_path)
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    input_size = X.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    fold = 1
    model_class = get_models(model_name)

    for train_index, test_index in kf.split(X):
        print(f"Processing fold {fold}/{n_splits}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = model_class(input_size=input_size, output_size=1)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, early_stopping_patience)
        y_true, y_pred_prob = test_model(model, test_loader, device)
        y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)
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
        print(
            f"Fold {fold}, Model: {model_name},"
            f"ACC: {acc:.4f}, "
            f"AUC: {auc_score:.4f}, "
            f"AUPR: {aupr:.4f}, "
            f"F1: {f1:.4f}, "
            f"Sn: {sn:.4f}, "
            f"Sp: {sp:.4f}"
        )
        fold += 1

    data_file_name = os.path.basename(data_path).split(".")[0]
    result_dir = os.path.join(results_base_path, model_name, data_file_name)
    os.makedirs(result_dir, exist_ok=True)
    results_output_path = os.path.join(result_dir, f"{data_file_name}_results.csv")
    save_results(results, results_output_path)

if __name__ == '__main__':
    MODEL_NAME = "TextRCNN"
    RESULTS_BASE_PATH = r"./data/results/ksu_Unreduced"
    data_paths = [
        # r"./data/undersample/KSU_Euclidean.csv",
        r"./data/undersample/KSU_Hamming.csv",
        # r"./data/undersample/KSU_Manhattan.csv",
        # r"./data/undersample/KSU_Minkowski.csv",
        # r"./data/undersample/KSU_Chebyshev.csv"
    ]

    for data_path in data_paths:
        print(f"Starting to process dataset: {data_path}")
        main(data_path, RESULTS_BASE_PATH, MODEL_NAME)
        print(f"Dataset {data_path} processing completed.\n")

