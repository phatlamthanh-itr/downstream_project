from downstream_clustering.clustering import plot_ecg_data
from downstream_clustering.cluster_strip2 import get_sample_from_hea
import os
import wfdb
path="data/strip2"

# record_path = "3392931/67508a48f9a08b000138bf08"
record_path = "399467/676b4a86d4ada800014e9097"

record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(path, record_path)) if '.dat' in file]
record_path = os.path.join(path, os.path.join(record_path, f"{record_ids[0]}"))
record = wfdb.rdrecord(record_path)
waveform = record.__dict__['p_signal']

waveform, flag_get_sample = get_sample_from_hea(record_path, waveform)
plot_ecg_data(waveform[:, [0, 1]])