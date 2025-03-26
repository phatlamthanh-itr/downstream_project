
import collections
from collections import Counter
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt 
from data.process.dataset_config import SAMPLING_RATE, DS_EVAL, NUM_SAMPLES_PER_FRAME
from wfdb.processing import resample_multichan
from data.process.preprocessing import butter_bandpass_filter, enforce_min_spacing, is_beat
from data.process.dataset_config import OFFSET_SAMPLE





def check(ecgpath="data/ecg_segmentation", DS_EVAL = DS_EVAL):
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]


    # Loop through records to create ecgs and labels
    for record_id in sorted(DS_EVAL):
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", str(record_id))
        # Get waveform in DS_EVAL only
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']
        people_annotation = wfdb.rdann(record_path, 'atr')
        AI_annotation = wfdb.rdann(record_path, 'beat')
        fs = record.fs
        org_sample = people_annotation.sample
        org_label = people_annotation.symbol

            # # Resample to 250Hz
            # if fs != SAMPLING_RATE:
            #     waveform, resampled_ann = resample_multichan(waveform, people_annotation, fs, SAMPLING_RATE)
            #     labels = resampled_ann.symbol
            #     sample = resampled_ann.sample

            # print("Org sample: ", org_sample[:20], len(org_sample))
            # print("Resampled sample: ", sample[:20], len(sample))
        people_sample = people_annotation.sample
        AI_sample = AI_annotation.sample
        print(people_sample[:100], "people sample", len(people_sample))
        print(AI_sample[:100], "AI sample", len(AI_sample))
        # print(np.mean(people_sample == AI_sample))
        print("==============")

            # print(people_sample, len(people_sample))
            # print(AI_sample, len(AI_sample))




if __name__ == "__main__":
    check()