import pandas as pd

disease_features_df = pd.read_csv('./data/B-datasets/feature_extraction/NEWDiseaseFeature.csv', header=None,usecols=range(2, 66))
drug_features_df = pd.read_csv('./data/B-datasets/feature_extraction/Drug_mol2vec.csv', header=0,usecols=range(2, 302))
association_numbers_df = pd.read_csv('./data/B-datasets/DrugDiseasedisAssociationNumber.csv')

drug_features_df.insert(0, 'drug', range(0, len(drug_features_df)))
drug_feature_names = [f'feature_{i}_x' for i in range(1, drug_features_df.shape[1])]
drug_features_df.columns = ['drug'] + drug_feature_names

disease_features_df.insert(0, 'disease', range(0, len(disease_features_df)))
disease_feature_names = [f'feature_{i}_y' for i in range(1, disease_features_df.shape[1])]
disease_features_df.columns = ['disease'] + disease_feature_names

merged_disease = association_numbers_df.merge(drug_features_df, on="drug", how="left")
print("After merging with disease features:")
print(merged_disease.head())
final_merged = merged_disease.merge(disease_features_df, on="disease", how="left")
print("After merging with drug features:")
print(final_merged.head())

reordered_columns = ['drug', 'disease'] + [col for col in final_merged.columns if 'feature' in col]
final_reordered = final_merged[reordered_columns]
final_reordered.to_csv('./data/original_samples/disAssociation.csv', index=False)
print(f"A total of {len(final_reordered)} disassociated drug-disease pairs have been generated.")