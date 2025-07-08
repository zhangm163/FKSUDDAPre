import pandas as pd

table1 = pd.read_csv('./data/original_samples/disAssociation.csv')
table2 = pd.read_csv('./data/undersample/disAssociation/disKSU_Hamming.csv')

key_columns = ['drug', 'disease']
merged = table1.merge(table2, on=key_columns, how='left', indicator=True)
table1_filtered = table1[merged['_merge'] == 'left_only']
table1_filtered = table1_filtered.sort_values(by='disease', ascending=True)

disease_counts = table1_filtered['disease'].value_counts().head(3)
print("Top 3 diseases with the highest case counts:")
print(disease_counts)

output_path = './data/candidate_drug/candidate_drug.csv'
table1_filtered.to_csv(output_path, index=False)
print(f"Data filtering completed. Results saved to {output_path}")

for i, (disease, count) in enumerate(disease_counts.items(), start=1):
    disease_data = table1_filtered[table1_filtered['disease'] == disease]
    disease_output_path = f'./data/candidate_drug/candidate_drug_top{i}_disease{disease}.csv'
    disease_data.to_csv(disease_output_path, index=False)
    print(f"Data for disease ID {disease} saved to {disease_output_path}")

disease_ids = [425, 214]
for disease_id in disease_ids:
    disease_data = table1_filtered[table1_filtered['disease'] == disease_id]
    disease_output_path = f'./data/candidate_drug/candidate_drug_disease{disease_id}.csv'
    disease_data.to_csv(disease_output_path, index=False)
    print(f"Data for disease ID {disease_id} saved to {disease_output_path}")


