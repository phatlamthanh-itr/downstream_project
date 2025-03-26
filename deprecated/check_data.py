import numpy as np
import collections
from collections import Counter





# train_data = np.load("data/ecg_segmentation/processed/train_data.npy")
# train_labels = np.load("data/ecg_segmentation/processed/train_labels.npy")
# val_labels = np.load("data/ecg_segmentation/processed/val_labels.npy")

train_labels = np.load("data/ecg_segmentation/processed/train_labels_subseq.npy")
val_labels = np.load("data/ecg_segmentation/processed/train_labels_subseq_80.npy")
print(train_labels.shape)
print(val_labels.shape)

# train_data = np.load("data/ecg_segmentation/processed/train_data_subseq.npy")
# val_data = np.load("data/ecg_segmentation/processed/val_data_subseq.npy")

# print(train_data.shape)
# print(val_data.shape)


# total_1 = 0
# total_0 = 0
# for sample in train_labels:
#     print(sample)
#     print('======')
#     total_1 += Counter(sample)[1]
#     total_0 += Counter(sample)[0]
# print("Total 1",total_1, "Total 0:", total_0)

# total_1 = 0
# total_0 = 0
# for sample in val_labels:
#     # print(Counter(sample))
#     total_1 += Counter(sample)[1]
#     total_0 += Counter(sample)[0]
# print(total_1, total_0)