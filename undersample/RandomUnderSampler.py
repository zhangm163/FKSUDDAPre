import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.utils import resample

group_sizes = [3423] * 5 + [1301]

def load_data():
    try:
        final_reordered = pd.read_csv('./data/original_samples/disAssociation.csv')
        print(final_reordered.head())
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        raise
    return final_reordered

def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df

def random_downsample(X, y, sampling_strategy):
    y = y.ravel()
    X_resampled = []
    y_resampled = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        if cls in sampling_strategy:
            target_count = sampling_strategy[cls]
            X_cls_downsampled, y_cls_downsampled = resample(
                X_cls, y_cls,
                replace=False,
                n_samples=target_count,
                random_state=42
            )
            X_resampled.append(X_cls_downsampled)
            y_resampled.append(y_cls_downsampled)
        else:
            X_resampled.append(X_cls)
            y_resampled.append(y_cls)
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    return X_resampled, y_resampled

def group_downsampling_with_random(df, group_sizes):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"Processing group: {group}")
        group_data = df[df['drug_group'] == group]
        feature_columns = [col for col in group_data.columns if 'feature' in col]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values
        group_data_diseases = group_data['disease'].values
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
        X_resampled, y_resampled = random_downsample(group_data_features, group_data_labels, sampling_strategy)
        print(f"Group: {group}, size after downsampling: {X_resampled.shape}")
        resampled_indices = []
        for label in label_counts.keys():
            indices = np.where(group_data_labels == label)[0]
            if sampling_strategy[label] < len(indices):
                resampled_indices.extend(np.random.choice(indices, sampling_strategy[label], replace=False))
            else:
                resampled_indices.extend(indices)
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = group_data_diseases[resampled_indices]
        group_data_resampled['drug'] = group_data['drug'].iloc[resampled_indices].values
        group_data_resampled['drug_group'] = group
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
        resampled_data.to_csv('./data/undersample/disAssociation/disRandomUnderSampler.csv', index=False)
        print("Downsampling complete. Data saved successfully.")
    except Exception as e:
        print(f"Error while saving file: {e}")

if __name__ == '__main__':
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    try:
        resampled_data, resampled_labels = group_downsampling_with_random(final_reordered, group_sizes)
        group_counts = resampled_data['drug_group'].value_counts()
        print("Number of samples per group after sampling:")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"An error occurred during downsampling: {e}")
        raise
