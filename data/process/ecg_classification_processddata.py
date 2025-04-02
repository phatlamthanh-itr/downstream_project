import numpy as np

def labeling_added_noise(len_label = 1800):
    return [0 if (idx//18) % 2 == 0 else 1 for idx in range(len_label)]

def labeling_MIT_noise():
    all_labels = [] 
    
    for _ in range(12):
        labels = np.zeros(180)
        for i in range(30, 180, 24):
            labels[i:i+11] = 1  
        all_labels.append(labels) 
    all_labels = np.array(all_labels)
    all_labels = all_labels.reshape(-1)
    all_noise = np.ones(540)
    all_labels = np.concatenate((all_labels, all_noise), axis = 0)

label = labeling_MIT_noise()
print(np.sum(label))
