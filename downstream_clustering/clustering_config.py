CLUSTER_MODEL = "KMEANS" # KMEANS | MINIBATCHKMEANS | "SVC"
CLUSTER_RETRAIN = False
REDUCE_DIMENSIONS = "PCA" # TSNE | PCA |
PATH_SAVE_DATA_REDUCED = "./data/ecg_clustering/processed/"
RECORD_IDS_TRAIN = ['119e_6', '118e_6'] # ['119e06', '119e24', '119e18', '118e24', '118e06', '118e_6', 'bw', 'ma', '118e18', 'em', '118e12', '119e_6', '118e00', '119e00', '119e12']
RECORD_IDS_TEST = ['119e_6', '119e00', '119e24']
NAME_PTH_FILE = "train_subseq"