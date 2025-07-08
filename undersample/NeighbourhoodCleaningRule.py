import pandas as pd
import numpy as np
from imblearn.under_sampling import NeighbourhoodCleaningRule
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

group_sizes = [3423] * 5 + [1301]

def load_data():
    try:
        final_reordered = pd.read_csv('./data/original_samples/disAssociation.csv')
        print(final_reordered.head())
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        raise
    return final_reordered

def UnderNCR(req, X, y, meta):
    n_neighbors = int(req.form.get(f'NCR_n_neighbors'))
    sampling_strategy = req.form.get(f'NCR_sampling_strategy')
    ncr = NeighbourhoodCleaningRule(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors)
    resampled_data, resampled_labels = ncr.fit_resample(X, y)
    selected_indices = ncr.sample_indices_
    meta_resampled = meta.iloc[selected_indices].reset_index(drop=True)
    return resampled_data, resampled_labels, meta_resampled

def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df


def group_downsampling_with_NCR(df, group_sizes):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"Processing group: {group}")
        group_data = df[df['drug_group'] == group]
        feature_columns = [col for col in group_data.columns if 'feature' in col]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values.astype(int)
        group_data_diseases = group_data['disease'].values
        unique_labels = np.unique(group_data_labels)
        class MockRequest:
            def __init__(self):
                self.form = {
                    'NCR_n_neighbors': '6',
                    'NCR_sampling_strategy': 'auto',
                }
        request = MockRequest()
        X_resampled, y_resampled, meta_resampled = UnderNCR(request, group_data_features, group_data_labels,
                                                            group_data[['drug', 'disease']])
        print(f"Group: {group}, size after downsampling: {X_resampled.shape}")
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = meta_resampled['disease']
        group_data_resampled['drug_group'] = group
        group_data_resampled['drug'] = y_resampled
        group_data_resampled = group_data_resampled[['drug', 'disease'] + feature_columns + ['drug_group']]
        if len(group_data_resampled) > size:
            group_data_resampled = group_data_resampled.sample(n=size, random_state=42)
        resampled_data.append(group_data_resampled)
        resampled_labels.extend(y_resampled)
    final_resampled_data = pd.concat(resampled_data)
    final_resampled_labels = np.array(resampled_labels)
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        group_data = final_resampled_data[final_resampled_data['drug_group'] == group]
        if len(group_data) != size:
            print(f"Warning: Group '{group}' has {len(group_data)} samples, expected {size}.")
            if len(group_data) < size:
                original_group_data = df[df['drug_group'] == group]
                additional_samples = original_group_data.sample(n=size - len(group_data), replace=True, random_state=42)
                group_data = pd.concat([group_data, additional_samples])
                final_resampled_data = pd.concat(
                    [final_resampled_data[final_resampled_data['drug_group'] != group], group_data])
    return final_resampled_data, final_resampled_labels

def save_resampled_data(resampled_data):
    try:
        print("Saving downsampled data...")
        resampled_data = resampled_data.drop(columns=['drug_group'])
        resampled_data['drug'] = resampled_data['drug'].astype(int)
        resampled_data['disease'] = resampled_data['disease'].astype(int)
        resampled_data = resampled_data.sort_values(by=['drug', 'disease'], ascending=True)
        resampled_data.to_csv('./data/undersample/disAssociation/disNeighbourhoodCleaningRule.csv', index=False)
        print("Downsampling complete. Data saved successfully.")
    except Exception as e:
        print(f"Error while saving file: {e}")

if __name__ == '__main__':
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    try:
        resampled_data, resampled_labels = group_downsampling_with_NCR(final_reordered, group_sizes)
        group_counts = resampled_data['drug_group'].value_counts()
        print("Number of samples per group after sampling:")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"An error occurred during downsampling: {e}")
        raise
