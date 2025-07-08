import pandas as pd
import itertools


df_related = pd.read_csv('/./data/B-datasets/DrugDiseaseAssociationNumber.csv')
all_drugs = df_related['drug'].unique()
all_diseases = df_related['disease'].unique()

all_combinations = set(itertools.product(all_drugs, all_diseases))
related_pairs_set = set(tuple(x) for x in df_related.values)
unrelated_pairs = all_combinations - related_pairs_set
df_unrelated = pd.DataFrame(list(unrelated_pairs), columns=['drug', 'disease'])
df_unrelated = df_unrelated.drop_duplicates().reset_index(drop=True)
df_unrelated_sorted = df_unrelated.sort_values(by=['drug', 'disease']).reset_index(drop=True)
df_unrelated_sorted.to_csv('./data/B-datasets/DrugDiseasedisAssociationNumber.csv',index=False)

