import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv('./data/undersample/KSU_Hamming.csv')
X = data.iloc[:, 3:303]
y = data['drug']

k_values = range(70, 200, 10)

for k in k_values:
    print(f"Selecting {k} features using SelectKBest with f_classif...")
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_features}")
    X_selected_df = X.iloc[:, selected_features]
    selected_data = X_selected_df.copy()
    selected_data['drug'] = data['drug']
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']
    disease_columns = [col for col in data.columns if col.endswith('_y')]
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if col not in ['drug', 'disease', 'label']]]
    
    output_file = f'./data/after_dimension_reduction/{k}/f_classif_KSU_Hamming{k}.csv'
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    selected_data.to_csv(output_file, index=False)
    print(f"Feature-selected data has been saved to '{output_file}'")

