import sys
import os
import pandas as pd
import numpy as np

from atom_embedding_generator import bert_atom_embedding

# Configuration Parameters
input_file_path = './data/B-datasets/DrugInformation.csv'
output_dir = './data/B-datasets/feature_extraction/'
output_file_name = 'k_bert_embedding.csv'
EMBEDDING_DIM = 768
PRETRAIN_MODEL = 'pretrain_k_bert_epoch_7.pth'

if not os.path.exists(input_file_path):
    print(f"Error: Input file not found at {input_file_path}")
    sys.exit(1)

print(f"Reading data from {input_file_path}...")
dataset = pd.read_csv(input_file_path, index_col=None)
smiles_list = dataset['smiles'].values.tolist()

pretrain_features_list = []


print("Starting K-BERT embedding extraction...")
for i, smiles in enumerate(smiles_list):
    # Print progress (every 100 iterations to avoid clutter)
    if (i + 1) % 100 == 0:
        print(f"Processing: {i + 1}/{len(smiles_list)}")

    try:
        # Call K-BERT to extract features
        h_global, g_atom = bert_atom_embedding(smiles, pretrain_model=PRETRAIN_MODEL)

        # Ensure format is a list
        if hasattr(h_global, 'tolist'):
            h_global = h_global.tolist()

        pretrain_features_list.append(h_global)

    except Exception as e:
        # Print warning but continue execution
        print(f"Warning: Failed at index {i} (SMILES: {smiles}). Error: {e}")
        # Fill with NaNs to maintain consistent dimensions
        pretrain_features_list.append([np.nan] * EMBEDDING_DIM)


print("Constructing DataFrame...")

# Generate column names: pretrain_feature_1 to pretrain_feature_768
feature_cols = [f'pretrain_feature_{i + 1}' for i in range(EMBEDDING_DIM)]

# Convert list to DataFrame
features_df = pd.DataFrame(pretrain_features_list, columns=feature_cols)

# Horizontal Concatenation
dataset = pd.concat([dataset, features_df], axis=1)


# Remove rows where extraction failed (check if the first feature is NaN)
initial_len = len(dataset)
dataset = dataset.dropna(subset=['pretrain_feature_1'])
print(f"Removed {initial_len - len(dataset)} rows due to extraction errors.")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file_name)

# Save to CSV
dataset.to_csv(output_path, index=False)
print(f"Done! Saved result to: {output_path}")