import os
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import cuml.accel
cuml.accel.install()
# from cuml.manifold import TSNE as cuTSNE
import joblib
from .clustering_config import *

def reduce_dimensions(data, out_dim, model_reduce = REDUCE_DIMENSIONS):
    assert (model_reduce == "TSNE" or model_reduce == "PCA" or model_reduce == "UMAP")
    if model_reduce == "TSNE":
        if data.shape[1] > out_dim:
            print("TSNE Reducing dimensions...")
            tsne = TSNE(n_components=2, random_state=42)
            # tsne = cuTSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(data)
            print(f'Reduced dimensions from {data.shape} dimentions to {reduced_data.shape} dimentions sucessfully!')
        else:
            reduced_data = data
        np.save(os.path.join(PATH_SAVE_DATA_REDUCED, "reduced_data_tsne.npy"), reduced_data)
        return reduced_data
    
    if model_reduce == "PCA":
        if data.shape[1] > out_dim:
            print("PCA Reducing dimensions...")
            pca = PCA(n_components=2, random_state=42)
            reduced_data = pca.fit_transform(data)
            print(f'Reduced dimensions from {data.shape} dimentions to {reduced_data.shape} dimentions sucessfully!')
        else:
            reduced_data = data
        np.save(os.path.join(PATH_SAVE_DATA_REDUCED, "reduced_data_pca.npy"), reduced_data)
        return reduced_data
    
    if model_reduce == "UMAP":
        if data.shape[1] > out_dim:
            print("UMAP Reducing dimensions...")
            umap_model = umap.UMAP(n_components=out_dim, random_state=42)
            reduced_data = umap_model.fit_transform(data)
            print(f'Reduced dimensions from {data.shape} dimentions to {reduced_data.shape} dimentions sucessfully!')
        else:
            reduced_data = data
        np.save(os.path.join(PATH_SAVE_DATA_REDUCED, "reduced_data_umap.npy"), reduced_data)
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
                    color=colors(i), label=f'Cluster {label}', alpha=0.6)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def clustering(rebar_model, data, save_path = "./data/ecg_clustering/processed/", k = None, reencode = False):
    #Using REBAR model to encode data
    if reencode or not os.path.exists(os.path.join(save_path, "encoded_data_clustering.pth")):
        data_encoded = rebar_model.encode(data)
        torch.save(data_encoded, os.path.join(save_path, "encoded_data_clustering.pth"), pickle_protocol=4)
        print(f"Encode and save data to {os.path.join(save_path, 'encoded_data_clustering.pth')} successfully!")
    else:
        data_encoded = torch.load(os.path.join(save_path, "encoded_data_clustering.pth"), weights_only=False)
        print(f"Load data from path {os.path.join(save_path, 'encoded_data_clustering.pth')} successfully!")

    # print(data_encoded.shape) (2700, 2500, 320)
    data_encoded = data_encoded.reshape(-1, 320) # data_encoded.shape = (6750000, 320)
    scaler = StandardScaler()
    data_encoded = scaler.fit_transform(data_encoded)
    data_encoded = data_encoded[:100000, :]
    print("data encode shape: ", data_encoded.shape)
    data_encoded = reduce_dimensions(data_encoded, out_dim=2)

    if CLUSTER_MODEL == "KMEANS":
        if CLUSTER_RETRAIN == False and os.path.exists('./experiments/out/cluster/kmeans/clustering_pipeline.pkl'):
            pipeline = joblib.load('./experiments/out/cluster/kmeans/clustering_pipeline.pkl')
            print("Load KMeans model success!")
        else:
            print("Fitting Kmeans model...")
            pipeline = Pipeline([   
                ('kmeans', KMeans(n_clusters=2, random_state=42, n_init= 10)) 
            ])
            pipeline.fit(data_encoded)
            os.makedirs('./experiments/out/cluster/kmeans', exist_ok=True)
            joblib.dump(pipeline, './experiments/out/cluster/kmeans/clustering_pipeline.pkl')
            print("Fit and save KMEANs model success")

    elif CLUSTER_MODEL == "MINIBATCHKMEANS":
        if CLUSTER_RETRAIN == False and os.path.exists('./experiments/out/cluster/minibatchkmeans/clustering_pipeline.pkl'):
            pipeline = joblib.load('./experiments/out/cluster/minibatchkmeans/clustering_pipeline.pkl')
            print("Load MiniBatch KMeans model success!")
        else:
            print("Fitting MiniBatch KMEANs model...")
            pipeline = Pipeline([   
                ('kmeans', MiniBatchKMeans(n_clusters=2, random_state=42, n_init= 10, batch_size=1024)) 
            ])
            pipeline.fit(data_encoded)
            os.makedirs('./experiments/out/cluster/minibatchkmeans', exist_ok=True)
            joblib.dump(pipeline, './experiments/out/cluster/minibatchkmeans/clustering_pipeline.pkl')
            print("Fit and save MiniBatch KMEANs model success")

    data_plot = data_encoded[:100000, :]
    print("data plot shape: ", data_plot.shape)
    labels = pipeline.predict(data_plot)
    # print("Calculating silhouette sorce...")
    # score = silhouette_score(data_encoded, labels)
    # print("Silhouette Score:", score)
    plot_clustering(data=data_plot, labels=labels)



