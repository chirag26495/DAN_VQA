import numpy as np
# from sklearn.neighbors import KDTree
# from sklearn.neighbors import DistanceMetric
# from sklearn.cluster import KMeans
# from tqdm import tqdm
# import time
# import sys

sample_dict = {0: (0, 53588), 
                1: (53588, 107176),
                2: (107176, 160764),
                3: (160764, 214354)}


vqa_features = np.load('../data/val_vqa_features.npy', allow_pickle=True).tolist()
indices, features = vqa_features['vqa_idx'], vqa_features['features_fc1']
indices = np.array(indices)

dataset_sz = len(features)

print(f"---------Size of dataset: {dataset_sz}----------")

distance_matrix = np.zeros(shape=(dataset_sz, 1500))
neighbour_matrix = np.zeros(shape=(dataset_sz, 1500)).astype(np.int_)

for i in range(4) :
    start_idx, ind_idx = sample_dict[i]
    distances_sub = np.load(f'../models/fixed_val_neighbor_distance_matrix_{i}.npy')
    neighbors_sub = np.load(f'../models/fixed_val_neighbor_matrix_{i}.npy')
    
    print(f"------------------i={i}---------------")
    for j in range(len(neighbors_sub)) :
        neighbour_matrix[indices[start_idx + j]] = neighbors_sub[j]
        distance_matrix[indices[start_idx + j]] = distances_sub[j]

np.save(f'../models/fixed_val_neighbor_distance_matrix.npy', distance_matrix)
np.save(f'../models/fixed_val_neighbor_matrix.npy', neighbour_matrix)
