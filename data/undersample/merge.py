import pandas as pd

df_related = pd.read_csv('./data/original_samples/Association.csv')
df_unrelated = pd.read_csv('./data/undersample/disAssociation/disKSU_Hamming.csv')
df_related['label'] = 1
df_unrelated['label'] = 0

df_related.columns = df_related.columns.str.strip()
df_unrelated.columns = df_unrelated.columns.str.strip()
print("Columns in df_related:", df_related.columns)
print("Columns in df_unrelated:", df_unrelated.columns)

drug_columns = [col for col in df_related.columns if col.endswith('_x')]
disease_columns = [col for col in df_related.columns if col.endswith('_y')]
print("Drug columns:", drug_columns)
print("Disease columns:", disease_columns)

df = pd.concat([df_related, df_unrelated], ignore_index=True)
df = df[['drug', 'disease', 'label'] + drug_columns + disease_columns]
print(df.head())
print("Columns in merged data:", df.columns)

output_file = '/home/zhangcy/file/FKSUDDAPre/data/undersample/KSU_Hamming.csv'
df.to_csv(output_file, index=False)
print("Merged data saved to", output_file)

