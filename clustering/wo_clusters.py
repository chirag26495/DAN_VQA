import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import sys

print("------------Imports Completed-------------")


sample_dict_sub = sample_dict = {0  : ( 0 , 13397 ),
                1  : ( 13397 , 26794 ),
                2  : ( 26794 , 40191 ),
                3  : ( 40191 , 53588 ),
                4  : ( 53588 , 66985 ),
                5  : ( 66985 , 80382 ),
                6  : ( 80382 , 93779 ),
                7  : ( 93779 , 107176 ),
                8  : ( 107176 , 120573 ),
                9  : ( 120573 , 133970 ),
                10  : ( 133970 , 147367 ),
                11  : ( 147367 , 160764 ),
                12  : ( 160764 , 174161 ),
                13  : ( 174161 , 187558 ),
                14  : ( 187558 , 200955 ),
                15  : ( 200955 , 214352 )}


sample_dict = {0: (0, 53588), 
                1: (53588, 107176),
                2: (107176, 160764),
                3: (160764, 214354)}


# Arguments passed
script_index = int(sys.argv[1])
index_range = sample_dict[script_index]
n_elements = index_range[1] - index_range[0]
print("Script Index: ", script_index, "\tRange: ", index_range, "\tn_elements: ", n_elements)



vqa_features = np.load('../data/val_vqa_features.npy', allow_pickle=True).tolist()
indices, features = vqa_features['vqa_idx'], vqa_features['features_fc1']
indices = np.array(indices)

dataset_sz = len(features)

print(f"---------Size of dataset: {dataset_sz}----------")

metric = DistanceMetric.get_metric('minkowski', p=2)

print("---------------Making Tree----------------")

tree = KDTree(features)

print("--------------Tree Complete---------------")

distance_matrix = np.zeros(shape=(n_elements, 1500))
neighbour_matrix = np.zeros(shape=(n_elements, 1500)).astype(np.int_)


progress_bar = tqdm(total=dataset_sz, desc="Progress", unit="iteration", leave=True)

for i in range(n_elements) :
    t_0 = time.time()
    q_distances, q_indices = tree.query([features[index_range[0] + i]], 1500)
    t_1 = time.time()
    distance_matrix[i] = q_distances
    neighbour_matrix[i] = indices[q_indices]
    
    # print(f'query time:{t_1 - t_0}, k_means_1_time:{t_2 - t_1}, k_means_fitpredict:{t_3 - t_2}, computre_distances:{t_4 - t_3}, sort:{t_5 - t_4}, other:{t_6 - t_5}')
    
    progress_bar.update(1)
    etc = int(progress_bar.format_dict['elapsed'] * (n_elements - i - 1) / (i + 1))
    progress_bar.set_description(f"Progress (ETC: {etc//(3600*24)}d {(etc // 3600) % 24}h {(etc%3600)//60}min {etc%60}sec)")
    if i % 100 == 0 :
        print((i / n_elements) * 100)


np.save(f'neighbor_distance_matrix_{script_index}.npy', distance_matrix)
np.save(f'neighbor_matrix_{script_index}.npy', neighbour_matrix)

print("File Saved")