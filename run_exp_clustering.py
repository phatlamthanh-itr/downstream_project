import os
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from utils.utils import printlog
from utils.utils import import_model
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
import argparse
from utils.utils import printlog, load_data, import_model, init_dl_program
from downstream_clustering.clustering import clustering

all_expconfigs = {**allrebar_expconfigs}
data_path = "./data/ecg_clustering/processed/"
save_path = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str, required=True)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    args = parser.parse_args()

    # selecting config according to arg
    config = all_expconfigs[args.config]
    config.set_rundir(args.config)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", config.data_name, config.run_dir, "checkpoint_best.pkl"))):
        train_data, _, val_data, _, _, _ = load_data(config = config, data_type = "fullts")
        model = import_model(config, train_data=train_data, val_data=val_data)
        model.fit()

    # data = np.load(os.path.join(data_path, f"all_ecgs_subseq.npy"))
    data = np.load(os.path.join(data_path, f"all_ecgs_subseq_strip2.npy"))
    data = data[:4000,:, [0, 1]] # chon chanel thu 0 va 1
    config.set_inputdims(data.shape[-1])
    rebar_model = import_model(config, reload_ckpt = True)
    #os.system('cls' if os.name == 'nt' else 'clear')
    clustering(rebar_model, data= data)