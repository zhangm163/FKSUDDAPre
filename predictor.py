import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import torch
from tqdm import tqdm
from features import MolSentence, mol2alt_sentence 
from decision_tree_model import DecisionTreeModel
from random_forest_model import RandomForestModel
from xgboost_model import XGBoostModel
from feature_selector import FeatureSelector

MODEL_PATH = "model_300dim.pkl"
word2vec_model = word2vec.Word2Vec.load(MODEL_PATH)
RADIUS = 1
UNCOMMON = "UNK"

class PredictionThread(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, predictor, features, drug_ids, disease_ids):
        super().__init__()
        self.predictor = predictor
        self.features = features
        self.drug_ids = drug_ids
        self.disease_ids = disease_ids

    def run(self):
        try:
            results = []
            total = len(self.drug_ids)

            batch_size = 100
            for i in range(0, total, batch_size):
                batch_features = self.features[i:i + batch_size]
                probs, msg = self.predictor.predict_proba(batch_features)

                if probs is None:
                    self.error.emit(f"Prediction failed: {msg}")
                    return

                for j in range(len(probs)):
                    idx = i + j
                    results.append({
                        'drug_id': self.drug_ids[idx],
                        'disease_id': self.disease_ids[idx],
                        'probability': probs[j],
                        'label': 'Positive' if probs[j] > 0.5 else 'Negative'
                    }) 
                self.progress.emit(min(i + batch_size, total))
            self.finished.emit(results, None)
        except Exception as e:
            self.error.emit(f"Prediction thread error: {str(e)}")

class EnsemblePredictor:
    def __init__(self, model_path=None, feature_selector_params=None):
        self.models = {}  
        self.feature_selector_params = feature_selector_params or {} 
        if model_path:
            self.load_models(model_path)

    def load_models(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        try:
            model_dict = joblib.load(model_path)
            if not isinstance(model_dict, dict):
                raise FileNotFoundError(f"Model file does not exist: {model_path}")
            self.models = model_dict
            print("Loaded model dictionary: ", self.models)

            for model_name, content in self.models.items():
                print(f"Loading model {model_name} ...")
                model = content['model']
                feature_selector = self.feature_selector_params.get(model_name, None)
                if feature_selector:
                    print(f"Setting feature selector for model {model_name}, selected {feature_selector} features")
                if not isinstance(model, (DecisionTreeModel, RandomForestModel, XGBoostModel)):
                    raise ValueError(f"Model {model_name} has incorrect type: {type(model)}")
                print(f"Model {model_name} loaded successfully")

        except Exception as e:
            raise ValueError(f"Failed to load models: {str(e)}")

    def predict_proba(self, X_new):
        if not self.models:
            return None, "No models loaded"
        all_probs = []
        error_msgs = []
        for model_name, content in self.models.items():
            model = content['model']
            feature_selector = self.feature_selector_params.get(model_name, None)
            if feature_selector:
                print(f"Applying feature selector for ({model_name}), selecting {feature_selector} features")
                X_model = X_new.iloc[:, :feature_selector]
            else:
                X_model = X_new

            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_model)
                    y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                elif hasattr(model, 'predict'):
                    y_pred = model.predict(X_model)
                    y_prob = self._normalize_predictions(y_pred)
                else:
                    raise ValueError(f"Model {model_name} does not support prediction")

                all_probs.append(y_prob)
            except Exception as e:
                error_msgs.append(f"{model_name} prediction failed: {str(e)}")

        if not all_probs:
            return None, "All models failed to predict:\n" + "\n".join(error_msgs)
        return np.mean(all_probs, axis=0), ("\n".join(error_msgs) if error_msgs else None)


class DrugDiseasePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drug-Disease Association Prediction System")
        self.setGeometry(100, 100, 1200, 800)
        self.test_data_path = ""
        self.ensemble_model_path = ""
        self.predictor = None
        self.word2vec_model = None
        self.current_results = None

        # Define feature selection parameters for each model (number of features selected during training)
        self.feature_selector_params = {
            'DecisionTree': 140,  # DecisionTree uses first 140 features
            'RandomForest': 90,  # RandomForest uses first 90 features
            'XGBoost': 160,  # XGBoost uses first 160 features
        }

        self.init_ui()
        self.load_word2vec_model()


    def init_ui(self):
        """Initialize user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        self.results_table = self.create_results_table()
        main_layout.addWidget(self.results_table)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        self.setStyleSheet(""" 
            QPushButton { padding: 8px; font-size: 14px; min-width: 120px; }
            QLabel { font-size: 14px; }
            QProgressBar { height: 20px; }
        """)

    def create_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)

        self.model_btn = QPushButton("Select Model")
        self.model_btn.clicked.connect(self.select_model)
        self.model_label = QLabel("No model selected")
        layout.addWidget(self.model_btn)
        layout.addWidget(self.model_label)

        self.data_btn = QPushButton("Select Data")
        self.data_btn.clicked.connect(self.select_data)
        self.data_label = QLabel("No data selected")
        layout.addWidget(self.data_btn)
        layout.addWidget(self.data_label)

        self.predict_btn = QPushButton("Start Prediction")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        return panel

    def create_results_table(self):
        """Create results table"""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Drug ID', 'Disease ID', 'Probability', 'Label'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return table

    def load_word2vec_model(self):
        """Load pre-trained Word2Vec model"""
        try:
            self.word2vec_model = word2vec.Word2Vec.load("model_300dim.pkl")
            print("Word2Vec model loaded successfully")
        except Exception as e:
            print(f"Failed to load Word2Vec model: {str(e)}")

    def select_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Pickle Files (*.pkl)")
        if model_file:
            self.ensemble_model_path = model_file
            self.model_label.setText(os.path.basename(model_file))
            self.load_ensemble_model(model_file)

    def load_ensemble_model(self, model_path):
        try:
            self.predictor = EnsemblePredictor(model_path, self.feature_selector_params)
            self.status_label.setText("Model loaded successfully, ready for prediction")
            self.predict_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Model loaded successfully, ready for prediction")
            self.predict_btn.setEnabled(False)

    def select_data(self):
        data_file, _ = QFileDialog.getOpenFileName(self, "Select Test Data", "", "CSV Files (*.csv)")
        if data_file:
            self.test_data_path = data_file
            self.data_label.setText(os.path.basename(data_file))

    def start_prediction(self):
        if not self.test_data_path or not self.predictor:
            QMessageBox.warning(self, "Warning", "Please select data and model files")
            return

        try:
            test_data = pd.read_csv(self.test_data_path)
            drug_smiles = test_data['smiles'].values
            disease_ids = test_data['disease'].values
            drug_ids = test_data['drug'].values
            X_new_feats = []
            invalid_count = 0
            for smiles in drug_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    try:
                        sentence = mol2alt_sentence(mol, radius=RADIUS)
                        vector = np.mean([self.word2vec_model.wv[token] if token in self.word2vec_model.wv else
                                          self.word2vec_model.wv[UNCOMMON]
                                          for token in sentence], axis=0)
                        X_new_feats.append(vector)
                    except Exception as e:
                        print(f"Failed to process molecule: {smiles} Error: {str(e)}")
                        X_new_feats.append(np.zeros(self.word2vec_model.vector_size))
                        invalid_count += 1
                else:
                    print(f"Invalid SMILES: {smiles}")
                    X_new_feats.append(np.zeros(self.word2vec_model.vector_size))
                    invalid_count += 1

            print(f"Number of invalid or failed molecules: {invalid_count}")
            disease_feats = test_data.iloc[:, -64:].values
            X_new_feats = pd.DataFrame(X_new_feats)
            feature_selector = self.feature_selector_params.get('DecisionTree', None)
            if feature_selector:
                X_new_feats = X_new_feats.iloc[:, :feature_selector]
            X_all_feats = pd.concat([X_new_feats, pd.DataFrame(disease_feats)], axis=1)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data: {str(e)}")
            return

        self.predict_thread = PredictionThread(self.predictor, X_all_feats, drug_ids, disease_ids)
        self.predict_thread.finished.connect(self.on_prediction_finished)
        self.predict_thread.error.connect(self.on_prediction_error)
        self.predict_thread.progress.connect(self.on_prediction_progress)
        self.progress_bar.setVisible(True)
        self.predict_thread.start()

    def save_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "Error", f"Failed to load data: {str(e)}")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")

        if not save_path:
            return

        try:

            df = pd.DataFrame(self.current_results)
            if save_path.endswith('.csv'):
                df.to_csv(save_path, index=False)
            elif save_path.endswith('.xlsx'):
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path + '.csv', index=False)
            QMessageBox.information(self, "Success", f"Results saved to: {save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save: {str(e)}")

    def on_prediction_finished(self, results, error):
        self.progress_bar.setVisible(False)
        if error:
            QMessageBox.warning(self, "Error", error)
        else:
            self.update_results_table(results)

    def on_prediction_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "Error", error_msg)
    def on_prediction_progress(self, value):
        self.progress_bar.setValue(value)

    def update_results_table(self, results):
        self.current_results = results  
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(result['drug_id'])))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(result['disease_id'])))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result['probability']:.4f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(result['label']))
        self.save_btn.setEnabled(True)  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrugDiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
