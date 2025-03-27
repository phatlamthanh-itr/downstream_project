import os
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib

def reduce_dimensions(data, out_dim):
    if data.shape[1] > out_dim:
        print("Reducing dimensions...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(data)
        print(f'Reduced dimensions from {data.shape} dimentions to {reduced_data.shape} dimentions sucessfully!')
    else:
        reduced_data = data
    return reduced_data

def plot_clustering(data, labels, title='Clustering visualization with t-SNE'):
    reduced_data = reduce_dimensions(data, out_dim=2)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Tạo colormap dựa trên số lượng cụm
    colors = plt.cm.get_cmap('tab10', n_clusters)

    plt.figure(figsize=(8, 6))
    # Vẽ từng cụm với màu sắc riêng
    for i, label in enumerate(unique_labels):
        cluster_points = reduced_data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors(i), label=f'Cluster {label}', alpha=0.6, edgecolors='w')
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def clustering(rebar_model, data, save_path = "./data/ecg_clustering/processed/", k = None, reencode = False):

    if reencode or not os.path.exists(os.path.join(save_path, "encoded_data_clustering.pth")):
        data_encoded = rebar_model.encode(data)
        torch.save(data_encoded, os.path.join(save_path, "encoded_data_clustering.pth"), pickle_protocol=4)
        print(f"Encode and save data to {os.path.join(save_path, 'encoded_data_clustering.pth')} successfully!")
    else:
        data_encoded = torch.load(os.path.join(save_path, "encoded_data_clustering.pth"))
        print(f"Load data from path {os.path.join(save_path, 'encoded_data_clustering.pth')} successfully!")

    # print(data_encoded.shape) (2700, 2500, 320)
    data_encoded = data_encoded.reshape(-1, 320) # data_encoded.shape = (6750000, 320)

    if os.path.exists('./experiments/out/clustering_pipeline.pkl'):
        pipeline = joblib.load('./experiments/out/clustering_pipeline.pkl')
        print("Load KMeans model success!")
    else:
        print("Fitting model...")
        pipeline = Pipeline([   
            ('kmeans', KMeans(n_clusters=2, random_state=42)) 
        ])
        pipeline.fit(data_encoded)
        joblib.dump(pipeline, './experiments/out/clustering_pipeline.pkl')
        print("Fit and save KMEANs model success")

   
    labels = pipeline.predict(data_encoded)
    print(labels, len(labels))
    # print("Calculating silhouette sorce...")
    # score = silhouette_score(data_encoded, labels)
    # print("Silhouette Score:", score)
    plot_clustering(data=data_encoded, labels=labels)



