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

def plot_ecg_data(ecg_data, sampling_rate=250, title="ECG Data", xlabel="Time (s)", ylabel="Amplitude"):
    """
    Vẽ tín hiệu ECG với dữ liệu có dạng (n_samples, 2).
    
    Parameters:
      ecg_data: numpy array có shape (n_samples, 2) với n_samples là số điểm và 2 là số kênh.
      sampling_rate: Tần số lấy mẫu (Hz). Mặc định là 500.
      title: Tiêu đề của toàn bộ figure.
      xlabel: Nhãn trục x (áp dụng cho biểu đồ phía dưới).
      ylabel: Nhãn trục y (áp dụng cho từng biểu đồ).
    """
    if ecg_data.ndim != 2 or ecg_data.shape[1] != 2:
        raise ValueError(f"Dữ liệu phải có dạng (n_samples, 2) | {ecg_data.shape}")
    
    n_samples = ecg_data.shape[0]
    time_axis = np.linspace(0, n_samples / sampling_rate, n_samples)
    
    # Tạo figure với 2 subplot chia theo hàng
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # Vẽ Channel 1 trên subplot đầu tiên (phía trên)
    axes[0].plot(time_axis, ecg_data[:, 0], label="Channel 1", color="blue")
    axes[0].set_title("Channel 1")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)
    axes[0].legend()
    
    # Vẽ Channel 2 trên subplot thứ hai (phía dưới)
    axes[1].plot(time_axis, ecg_data[:, 1], label="Channel 2", color="red")
    axes[1].set_title("Channel 2")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True)
    axes[1].legend()
    
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_clustering(data, labels, title='Clustering visualization with t-SNE'):
    reduced_data = reduce_dimensions(data, out_dim=2)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Tạo colormap dựa trên số lượng cụm
    # colors = plt.cm.get_cmap('tab10', n_clusters)
    colors = ['blue', 'red']
    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    # Vẽ từng cụm với màu sắc riêng
    for i, label in enumerate(unique_labels):
        cluster_points = reduced_data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i], label=f'Cluster {label}', alpha=0.6)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)

    def on_click(event):
        if event.inaxes is not None:
            x_click, y_click = event.xdata, event.ydata
            # Tính khoảng cách từ điểm click đến tất cả các điểm trong không gian reduced_data
            distances = np.sqrt((reduced_data[:, 0] - x_click)**2 + (reduced_data[:, 1] - y_click)**2)
            idx = np.argmin(distances)
            print(f"Clicked on point index {idx}")
            # Gọi plot_ecg_data với dữ liệu ECG gốc của mẫu được chọn
            subseq = np.load("./data/ecg_clustering/processed/all_ecgs_subseq_strip2.npy")
            plot_ecg_data(subseq[idx][:, [0,1]], title=f"ECG for sample {idx}")
    
    # Kết nối sự kiện click với figure clustering
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

def clustering(rebar_model, data, save_path = "./data/ecg_clustering/processed/", k = None, reencode = False):
    #Using REBAR model to encode data
    name_pth_file = f"{NAME_PTH_FILE}.pth"
    if reencode or not os.path.exists(os.path.join(save_path, name_pth_file)):
        data_encoded = rebar_model.encode(data)
        torch.save(data_encoded, os.path.join(save_path, name_pth_file), pickle_protocol=4)
        print(f"Encode and save data to {os.path.join(save_path, name_pth_file)} successfully!")
    else:
        data_encoded = torch.load(os.path.join(save_path, name_pth_file), weights_only=False)
        print(f"Load data from path {os.path.join(save_path, name_pth_file)} successfully!")

    # print(data_encoded.shape) (2700, 2500, 320)
    # data_encoded = data_encoded.reshape(-1, 320) # data_encoded.shape = (6750000, 320)
    data_encoded = np.max(data_encoded, axis= 1) # data_encode.shape = (2700, 320)
    scaler = StandardScaler()
    data_encoded = scaler.fit_transform(data_encoded)

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
                ('kmeans', MiniBatchKMeans(n_clusters=2, random_state=42, n_init= 10, batch_size=384)) 
            ])
            pipeline.fit(data_encoded)
            os.makedirs('./experiments/out/cluster/minibatchkmeans', exist_ok=True)
            joblib.dump(pipeline, './experiments/out/cluster/minibatchkmeans/clustering_pipeline.pkl')
            print("Fit and save MiniBatch KMEANs model success")

    print("data plot shape: ", data_encoded.shape)
    labels = pipeline.predict(data_encoded)
    score = silhouette_score(data_encoded, labels)
    print("Silhouette Score:", score)
    plot_clustering(data=data_encoded, labels=labels)
    



