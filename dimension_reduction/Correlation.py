import os
import pandas as pd
data = pd.read_csv('./data/undersample/KSU_Hamming.csv')
print(f"Missing values in each column:\n{data.isnull().sum()}")
X = data.iloc[:, 3:303]
disease_columns = [col for col in data.columns if col.endswith('_y')]
y = data['drug']
correlations = X.corrwith(y)
abs_correlations = correlations.abs().sort_values(ascending=False)

k_values = range(70, 200, 10)

for k in k_values:
    print(f"Selecting {k} features using Correlation...")
    important_features = abs_correlations.index[:k]
    X_selected = X[important_features]
    selected_data = X_selected.copy()
    selected_data['drug'] = data['drug']
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if col not in ['drug', 'disease', 'label']]]

    output_file = f'./data/after_dimension_reduction/{k}/Correlation_KSU_Hamming{k}.csv'
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    selected_data.to_csv(output_file, index=False)
    print(f"Feature-selected data has been saved to '{output_file}'")

