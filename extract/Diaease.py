import pandas as pd

mesh_file = './data/B-datasets/MeSHFeatureGeneratedByDeepWalk.csv'
disease_file = './data/B-datasets/DiseaseFeature.csv'
mesh_data = pd.read_csv(mesh_file, header=None)  
disease_data = pd.read_csv(disease_file)

mesh_data.rename(columns={0: 'Disease'}, inplace=True) 
mesh_data.columns = ['Disease'] + [str(i) for i in range(1, mesh_data.shape[1])]
disease_data.rename(columns={disease_data.columns[1]: 'Disease'}, inplace=True)
mesh_data['Disease'] = mesh_data['Disease'].str.lower()
disease_data['Disease'] = disease_data['Disease'].str.lower()

filtered_data = mesh_data[mesh_data['Disease'].isin(disease_data['Disease'])]
filtered_data = filtered_data.set_index('Disease').reindex(disease_data['Disease']).reset_index()
filtered_data.insert(0, 'Index', range(1, len(filtered_data) + 1))

output_file = './data/B-datasets/feature_extraction/NEWDiseaseFeature.csv'
filtered_data.to_csv(output_file, index=False)