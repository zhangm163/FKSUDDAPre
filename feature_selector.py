from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

class FeatureSelector:
    def __init__(self, k=100):
        self.k = k
        self.selector = None
        self.feature_config = {}

    def fit(self, X, y):
        print(f"Fitting with X shape: {X.shape}, y shape: {y.shape}, k={self.k}")
        if self.k > X.shape[1]:
            self.k = X.shape[1]
            print(f"Adjusted k to {self.k} based on X features.")
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.selector.fit(X, y)
            self.selector.scores_ = np.nan_to_num(self.selector.scores_)
        selected_indices = self.selector.get_support(indices=True)
        return self

    def transform(self, X):
        if self.selector is None:
            raise ValueError("FeatureSelector has not been fitted yet.")
        print(f"Transforming with X shape: {X.shape}")
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def set_feature_config(self, model_name, selected_indices):
        self.feature_config[model_name] = selected_indices

    def get_selected_features(self, model_name):
        return self.feature_config.get(model_name, [])
