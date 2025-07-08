import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./data/undersample/KSU_Hamming.csv')
print(f"Missing values in each column:\n{data.isnull().sum()}")
X = data.iloc[:, 3:303]
disease_columns = [col for col in data.columns if col.endswith('_y')]
y = data['drug']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

k_values = range(90, 190, 10)

for k in k_values:
    print(f"Selecting {k} features using RFE...")
    selector = RFE(model, n_features_to_select=k, step=10)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_features}")
    X_selected = X.iloc[:, selected_features]
    selected_data = X_selected.copy()
    selected_data['drug'] = data['drug']
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if col not in ['drug', 'disease', 'label']]]

    output_file = f'./data/after_dimension_reduction/{k}/RFE_KSU_Hamming{k}.csv'
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    selected_data.to_csv(output_file, index=False)
    print(f"Feature-selected data has been saved to '{output_file}'")