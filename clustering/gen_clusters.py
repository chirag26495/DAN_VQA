import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

print("------------Imports Completed-------------")

vqa_features = np.load('../data/train_vqa_features.npy', allow_pickle=True).tolist()
indices, features = vqa_features['vqa_idx'], vqa_features['features_fc1']
indices = np.array(indices)

dataset_sz = len(features)

print(f"---------Size of dataset: {dataset_sz}----------")

metric = DistanceMetric.get_metric('minkowski', p=2)

print("---------------Making Tree----------------")

tree = KDTree(features)

print("--------------Tree Complete---------------")

ds_new = {}


progress_bar = tqdm(total=dataset_sz, desc="Progress", unit="iteration", leave=True)

for i in range(dataset_sz) :
    ds_new[i] = {}
    t_0 = time.time()
    q_distances, q_indices = tree.query([features[i]], 2000)
    t_1 = time.time()
    kmeans = KMeans(n_clusters=50, random_state=0)
    t_2 = time.time()
    labels_2k = kmeans.fit_predict(features[q_indices].squeeze())
    t_3 = time.time()
    
    cluster_centers = kmeans.cluster_centers_
    
    distances = metric.pairwise(cluster_centers, [features[i]])
    t_4 = time.time()
    sorted_cc_indices = np.argsort(distances.ravel())
    t_5 = time.time()
    
    for j in range(0, 4) :
        is_in = np.isin(labels_2k, sorted_cc_indices[j: j+1])
        penul_indices = np.where(is_in)
        final_indices = indices[penul_indices]
        ds_new[i][j] = final_indices
    
    for j in range(20, 24) :
        is_in = np.isin(labels_2k, sorted_cc_indices[j: j+1])
        penul_indices = np.where(is_in)
        final_indices = indices[penul_indices]
        ds_new[i][j] = final_indices
    t_6 = time.time()
    
    # print(f'query time:{t_1 - t_0}, k_means_1_time:{t_2 - t_1}, k_means_fitpredict:{t_3 - t_2}, computre_distances:{t_4 - t_3}, sort:{t_5 - t_4}, other:{t_6 - t_5}')
    
    progress_bar.update(1)
    etc = int(progress_bar.format_dict['elapsed'] * (dataset_sz - i - 1) / (i + 1))
    progress_bar.set_description(f"Progress (ETC: {etc//(3600*24)}d {etc//3600}h {(etc//60)%60}min {etc%60}sec)")


np.save('ds_new.npy', ds_new)