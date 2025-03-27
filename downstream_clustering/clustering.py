import os
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from utils.utils import printlog
from utils.utils import import_model
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def reduce_dimentions(data, out_dim):
    ...

def plot_clustering(data):
    ...

def clustering(rebar_model, data, save_path = "", k = None, reencode = False):

    if reencode or not os.path.exists(os.path.join(save_path, "encoded_data_clustering.pth")):
        data_encoded = rebar_model.encode(data)
        torch.save(data_encoded, os.path.join(save_path, "encoded_data_clustering.pth"), pickle_protocol=4)
    else:
        data_encoded = torch.load(os.path.join(save_path, "encoded_data_clustering.pth"))

    # print(data_encoded.shape) (2700, 2500, 320)

    pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('kmeans', KMeans(n_clusters=2, random_state=42)) 
    ])

    pipeline.fit(data_encoded)

    labels = pipeline.predict(data_encoded)

    score = silhouette_score(data_encoded, labels)
    print("Silhouette Score:", score)
    print("Cluster labels của 10 mẫu đầu tiên:", labels[:10])



