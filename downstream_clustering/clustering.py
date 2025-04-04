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
from sklearn.svm import OneClassSVM
import joblib
from .clustering_config import *
from downstream_classification.eval import calculate_accuracy, calculate_recall

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

def plot_ecg_data(ecg_data, sampling_rate=250, title="ECG Data", xlabel="Time (s)", ylabel="Amplitude"):
    if ecg_data.ndim != 2 or ecg_data.shape[1] != 2:
        raise ValueError(f"Dữ liệu phải có dạng (n_samples, 2) | {ecg_data.shape}")
    
    n_samples = ecg_data.shape[0]
    time_axis = np.linspace(0, n_samples / sampling_rate, n_samples)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(time_axis, ecg_data[:, 0], label="Channel 1", color="blue")
    axes[0].set_title("Channel 1")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].plot(time_axis, ecg_data[:, 1], label="Channel 2", color="red")
    axes[1].set_title("Channel 2")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True)
    axes[1].legend()
    
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_clustering(data, label_true, labels_pred, title='Clustering visualization with t-SNE'):
    reduced_data = reduce_dimensions(data, out_dim=2)
    
    correct0_mask = (labels_pred == label_true) & (label_true == 0)
    correct1_mask = (labels_pred == label_true) & (label_true == 1)
    incorrect_mask = (labels_pred != label_true)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if np.sum(correct0_mask) > 0:
        ax.scatter(reduced_data[correct0_mask, 0], reduced_data[correct0_mask, 1],
                   color='blue', label='True label 0 (correct)', alpha=0.6)
    if np.sum(correct1_mask) > 0:
        ax.scatter(reduced_data[correct1_mask, 0], reduced_data[correct1_mask, 1],
                   color='red', label='True label 1 (correct)', alpha=0.6)
    if np.sum(incorrect_mask) > 0:
        ax.scatter(reduced_data[incorrect_mask, 0], reduced_data[incorrect_mask, 1],
                   color='black', label='Incorrect prediction', alpha=0.6)
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()
    ax.grid(True)

    def on_click(event):
        if event.inaxes is not None:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((reduced_data[:, 0] - x_click)**2 + (reduced_data[:, 1] - y_click)**2)
            idx = np.argmin(distances)
            print(f"Clicked on point index {idx}")
            subseq = np.load(f"./data/ecg_clustering/processed/{NAME_PTH_FILE}.npy")
            plot_ecg_data(subseq[idx][:, [0, 1]], title=f"ECG for sample {idx}")
    
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

def clustering(rebar_model, data, save_path = "./data/ecg_clustering/processed/", k = None, reencode = False):
    #Using REBAR model to encode data
    name_pth_file = f"{NAME_PTH_FILE}.pth"
    if CLUSTER_RETRAIN:
        data_encoded = torch.load(os.path.join(save_path, name_pth_file), weights_only=False)
        data_encoded = np.max(data_encoded, axis= 1)
        print(f"Load data from path {os.path.join(save_path, name_pth_file)} for retrain successfully!")
    else:
        if reencode or not os.path.exists(os.path.join(save_path, name_pth_file)):
            data_encoded = rebar_model.encode(data)
            torch.save(data_encoded, os.path.join(save_path, name_pth_file), pickle_protocol=4)
            print(f"Encode and save data to {os.path.join(save_path, name_pth_file)} successfully!")
            data_encoded = np.max(data_encoded, axis= 1) 
        else:
            print("Loading...")
            data_encoded = torch.load(os.path.join(save_path, name_pth_file), weights_only=False)
            print(f"Loaded from {os.path.join(save_path, name_pth_file)}!")
            data_encoded = np.max(data_encoded, axis= 1) 
        
    # print(data_encoded.shape) (2700, 2500, 320)
    # data_encoded = data_encoded.reshape(-1, 320) # data_encoded.shape = (6750000, 320)
    # data_encoded = np.max(data_encoded, axis= 1) # data_encode.shape = (2700, 320)
    scaler = StandardScaler()
    data_encoded = scaler.fit_transform(data_encoded)
    test = True
    if CLUSTER_MODEL == "KMEANS":
        if CLUSTER_RETRAIN == False and os.path.exists('./experiments/out/cluster/kmeans/clustering_pipeline_118.pkl'):
            pipeline = joblib.load('./experiments/out/cluster/kmeans/clustering_pipeline_118.pkl')
            print("Load KMeans model success!")
        else:
            print("Fitting Kmeans model...")
            pipeline = Pipeline([   
                ('kmeans', KMeans(n_clusters=2, random_state=42, n_init= 10)) 
            ])
            pipeline.fit(data_encoded)
            os.makedirs('./experiments/out/cluster/kmeans', exist_ok=True)
            joblib.dump(pipeline, './experiments/out/cluster/kmeans/clustering_pipeline_118.pkl')
            print("Fit and save KMEANs model success")
            test = False

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
            test = False

    elif CLUSTER_MODEL == "SVC":
        if CLUSTER_RETRAIN == False and os.path.exists('./experiments/out/cluster/svc/clustering_pipeline.pkl'):
            pipeline = joblib.load('./experiments/out/cluster/svc/clustering_pipeline.pkl')
            print("Load SVC model success!")
        else:
            print("Fitting SVC model...")
            pipeline = Pipeline([   
                ('svc', OneClassSVM(kernel='rbf', gamma='scale')) 
            ])
            pipeline.fit(data_encoded)  
            os.makedirs('./experiments/out/cluster/svc', exist_ok=True)
            joblib.dump(pipeline, './experiments/out/cluster/svc/clustering_pipeline.pkl') 
            print("Fit and save SVC model success")

    labels_predicted = pipeline.predict(data_encoded)
    labels_predicted = 1 - labels_predicted # chuyen 0 -> 1 ; 1 -> 0
    score = silhouette_score(data_encoded, labels_predicted)
    print("Silhouette Score:", score)

    if test:
        labels_true = np.zeros(180)
        for i in range(30, 180, 24):
            labels_true[i:i+11] = 1
        print("Accuracy: ", calculate_accuracy(labels_true, labels_predicted))
        print("Recall: ", calculate_recall(labels_true, labels_predicted))
        plot_clustering(data=data_encoded, label_true=labels_true, labels_pred=labels_predicted)
    



