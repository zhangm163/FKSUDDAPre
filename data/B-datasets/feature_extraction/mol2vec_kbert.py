import pandas as pd
import os

mol2vec_file = './data/B-datasets/feature_extraction/Drug_mol2vec.csv'
kbert_file = './data/B-datasets/feature_extraction/k_bert_embedding.csv'

try:
    # Load Mol2Vec features (Column 0 + Columns 2-301)
    cols_mol2vec = [0] + list(range(2, 302))
    drug_features_mol2vec = pd.read_csv(mol2vec_file, header=0, usecols=cols_mol2vec)
    drug_features_mol2vec.rename(columns={drug_features_mol2vec.columns[0]: 'drug_id'}, inplace=True)

    # Load K-BERT features (Column 0 + Columns 3-770)
    cols_kbert = [0] + list(range(3, 771))
    drug_features_kbert = pd.read_csv(kbert_file, header=0, usecols=cols_kbert)
    drug_features_kbert.rename(columns={drug_features_kbert.columns[0]: 'drug_id'}, inplace=True)

    print("Files loaded successfully!")

except FileNotFoundError as e:
    print(f"Error: File not found. Please check paths: {e}")
    exit()

# Merge datasets on 'drug_id'
merged_df = pd.merge(drug_features_mol2vec, drug_features_kbert, on='drug_id', how='inner')
new_drug = merged_df

# Define output path
output_directory = './data/B-datasets'
output_filename = 'DrugFeature.csv'
full_output_path = os.path.join(output_directory, output_filename)

# Ensure directory exists and save
os.makedirs(output_directory, exist_ok=True)
new_drug.to_csv(full_output_path, index=False)

print("-" * 50)
print(f"Merged features saved to: '{full_output_path}'")
print(f"Data dimension (Rows, Cols): {new_drug.shape}")
print("\nPreview (First 5 rows):")
print(new_drug.head())
print("-" * 50)