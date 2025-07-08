import os
import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
import features
from tqdm import tqdm
from rdkit import RDLogger
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_NAME = "model_300dim.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
word2vec_model = word2vec.Word2Vec.load(MODEL_PATH)
RADIUS = 1
UNCOMMON = "UNK"

def smiles_to_sentence(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Failed to parse SMILES string: {smiles}")
        return None
    sentence = features.mol2alt_sentence(mol, radius)
    if not sentence:
        print(f"Warning: Generated molecular sentence is empty:  {smiles}")
    return features.MolSentence(sentence)

def sentence_to_vector(sentence, model, unseen=None):
    if sentence is None:
        print(f"Warning: Molecular sentence is None: {sentence}")
        return np.zeros(model.vector_size)
    vector = sum(model.wv[word] if word in model.wv else model.wv[unseen] for word in sentence.sentence)
    if np.all(vector == 0):
        print(f"Warning: The generated vector is all zeros: {sentence}")
    return vector

def extract_features_with_mol2vec(df, model, radius, unseen):
    tqdm.pandas(desc="Processing SMILES")
    df['mol_sentence'] = df['smiles'].progress_apply(lambda x: smiles_to_sentence(x, radius))
    vectors = df['mol_sentence'].apply(lambda x: sentence_to_vector(x, model, unseen))
    features = np.vstack(vectors.values)
    df.drop(columns=['mol_sentence'], inplace=True)
    return features

def save_features(features, original_data, output_path):
    feature_data = pd.DataFrame(features)
    original_fixed_columns = original_data.iloc[:, :2]
    combined_data = pd.concat([original_fixed_columns.reset_index(drop=True),
                               feature_data.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_data.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")

def main():
    input_file = "./data/B-datasets/DrugInformation.csv"
    output_file = "./data/B-datasets/feature_extraction/Drug_mol2vec.csv"
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist:  {input_file}")
        exit(1)
    print(f"Processing file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        if 'smiles' not in df.columns:
            print(f"Error: Column 'smiles' not found in file: {input_file}")
            exit(1)
        print(f"File contains {len(df)} records")
        features = extract_features_with_mol2vec(df, word2vec_model, RADIUS, UNCOMMON)
        save_features(features, df, output_file)
        print(f"Processing completed")
    except Exception as e:
        print(f"Error during processing: {e}")
        exit(1)
if __name__ == "__main__":
    main()