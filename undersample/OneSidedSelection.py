import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

group_sizes = [3423] * 5 + [1301]

def load_data():
    try:
        final_reordered = pd.read_csv('./data/original_samples/disAssociation.csv')
        print(final_reordered.head())
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        raise
    return final_reordered
def dist_nominal(instance, D):
    d = cdist(instance.reshape(1, -1), D.T, metric='hamming')[0]
    return d

def OSS_resample(data, labels, sampling_strategy):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label not in sampling_strategy:
            raise ValueError(f"Class label {label} in data is not present in sampling strategy.")
    data = data.T
    labels = labels.flatten()
    class_types = np.unique(labels)
    if set(sampling_strategy.keys()) != set(class_types):
        raise ValueError('Class labels in sampling strategy do not match data labels.')
    if data.shape[1] != len(labels):
        raise ValueError('Instance numbers in data and labels do not consistent.')
    class_data = {class_type: data[:, labels == class_type] for class_type in class_types}
    class_dist = {class_type: class_data[class_type].shape[1] for class_type in class_types}
    D = []
    DL = []
    for class_type in class_types:
        target_samples = sampling_strategy[class_type]
        if target_samples < class_dist[class_type]:
            K = class_data[class_type]
            indices = np.random.permutation(class_dist[class_type])
            id_to_keep = indices[:target_samples]
            D_temp = K[:, id_to_keep]
            DL_temp = np.array([class_type] * target_samples)
            D.append(D_temp)
            DL.append(DL_temp)
        else:
            D.append(class_data[class_type])
            DL.append(np.array([class_type] * class_dist[class_type]))
    D = np.hstack(D)
    DL = np.hstack(DL)
    final_class_dist = {class_label: np.sum(DL == class_label) for class_label in class_types}
    for class_label in class_types:
        if final_class_dist[class_label] != sampling_strategy[class_label]:
            print(
                f"Warning: Class '{class_label}' has {final_class_dist[class_label]} samples, expected {sampling_strategy[class_label]}.")
    return D.T, DL


def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df

def group_downsampling_with_OSS(df, group_sizes):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"Processing group: {group}")
        group_data = df[df['drug_group'] == group]
        feature_columns = [col for col in group_data.columns if 'feature' in col]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values
        group_data_diseases = group_data['disease'].values
        unique_labels = np.unique(group_data_labels)
        label_counts = Counter(group_data_labels)
        total_samples = sum(label_counts.values())
        sampling_strategy = {label: int(round(size * (count / total_samples))) for label, count in label_counts.items()}
        total_resampled = sum(sampling_strategy.values())
        if total_resampled > size:
            excess = total_resampled - size
            while excess > 0:
                for label in sorted(sampling_strategy, key=sampling_strategy.get, reverse=True):
                    if sampling_strategy[label] > 0:
                        sampling_strategy[label] -= 1
                        excess -= 1
                    if excess == 0:
                        break
        print(f"Group: {group}, original sample size: {group_data.shape[0]}")
        print(f"Label distribution: {label_counts}")
        X_resampled, y_resampled = OSS_resample(group_data_features, group_data_labels, sampling_strategy)
        print(f"Group: {group}, size after downsampling: {X_resampled.shape}")
        resampled_indices = []
        for label in unique_labels:
            indices = np.where(group_data_labels == label)[0]
            if sampling_strategy[label] < len(indices):
                resampled_indices.extend(np.random.choice(indices, sampling_strategy[label], replace=False))
            else:
                resampled_indices.extend(indices)
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = group_data_diseases[resampled_indices]
        group_data_resampled['drug_group'] = group
        group_data_resampled['drug'] = y_resampled
        group_data_resampled = group_data_resampled[['drug', 'disease'] + feature_columns + ['drug_group']]
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
        resampled_data.to_csv('./data/undersample/disAssociation/disOneSidedSelection.csv', index=False)
        print("Downsampling complete. Data saved successfully.")
    except Exception as e:
        print(f"Error while saving file: {e}")

if __name__ == '__main__':
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    try:
        resampled_data, resampled_labels = group_downsampling_with_OSS(final_reordered, group_sizes)
        group_counts = resampled_data['drug_group'].value_counts()
        print("Number of samples per group after sampling:")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"An error occurred during downsampling: {e}")
        raise
