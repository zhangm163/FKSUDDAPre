import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import time
import gc
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/undersample/KSU_Hamming.csv')
X = data.iloc[:, 3:303]
disease_columns = [col for col in data.columns if col.endswith('_y')]
y = data['drug']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)
print("Model training completed.")

subset_size = 20000
X_subset, _, y_subset, _ = train_test_split(X_scaled, y, test_size=0.9, stratify=y, random_state=42)
explainer = shap.TreeExplainer(model)
print("Starting SHAP values calculation...")
start_time = time.time()
shap_values = explainer.shap_values(X_subset)
end_time = time.time()
print(f"SHAP values calculated in {end_time - start_time:.2f} seconds.")
shap_importances = np.abs(shap_values).mean(axis=0).mean(axis=0)
print("SHAP importances calculated.")
print(f"SHAP importances: {shap_importances}")

k_values = range(90, 190, 10)

for k in k_values:
    print(f"Selecting {k} features using SHAP...")
    important_features = shap_importances.argsort()[::-1][:k]
    X_selected = X.iloc[:, important_features]
    selected_data = X_selected.copy()
    selected_data['drug'] = y
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if col not in ['drug', 'disease', 'label']]]

    output_file = f'./data/after_dimension_reduction/{k}/SHAP_KSU_Hamming{k}.csv'
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    selected_data.to_csv(output_file, index=False)
    print(f"Feature-selected data has been saved to '{output_file}'")

