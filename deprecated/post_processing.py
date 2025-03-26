import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt 
from data.process.dataset_config import SAMPLING_RATE, DS_EVAL, NUM_SAMPLES_PER_FRAME, FS_ORG
from data.process.preprocessing import enforce_min_spacing
from downstream.utils import plot_ecg_predictions, plot_ecg_predictions_random, get_peaks, rescale_peaks






# def post_processing(val_data, val_predictions, val_labels, ecgpath="data/ecg_segmentation", DS_EVAL = DS_EVAL):
#     # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
#     record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
#     # val_predictions, val_labels = reconstruct(val_predictions=val_predictions, val_labels=val_labels)  
#     val_labels = val_labels.reshape(len(DS_EVAL), -1)
#     val_predictions = val_predictions.reshape(len(DS_EVAL), -1)     # (patent, min(len(waveform)))


#     # Loop through records to create ecgs and labels
#     for record_id in sorted(record_ids):
#         record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
#         # Get waveform in DS_EVAL only
#         if int(record_id) in DS_EVAL: 
#             record = wfdb.rdrecord(record_path)
#             waveform = record.__dict__['p_signal']
#             annotation = wfdb.rdann(record_path, 'atr')
#             fs = record.fs
            
#             # Resample to 250Hz
#             if fs != SAMPLING_RATE:
#                 waveform, resampled_ann = resample_multichan(waveform, annotation, fs, SAMPLING_RATE)
#                 labels = resampled_ann.symbol
#                 sample = resampled_ann.sample

#             # Apply filters
#             filtered_waveform = butter_bandpass_filter(waveform)
    
#             # Normalize
#             filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
#             filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
#             # plt.figure(figsize=(10, 4))
#             # plt.plot(filtered_waveform[:2500, 0])
#             # plt.xlabel("Time Steps")
#             # plt.ylabel("Amplitude (Normalized)")
#             # plt.legend()
#             # plt.grid(True)
#             # plt.show()
#             # Create labels
#             padded_labels = np.zeros(len(waveform)).astype(int)
#             for i,l in enumerate(labels):
#                 if i==len(labels)-1:
#                     padded_labels[sample[i]:] = is_beat(l)
#                 else:
#                     padded_labels[sample[i] - OFFSET_SAMPLE // 2: sample[i] + OFFSET_SAMPLE // 2] = is_beat(l)

            
#             # Post-processing (make sure min_distance between two peaks greater than 0.4 s)
#             # padded_labels = enforce_min_spacing(padded_labels=padded_labels, min_distance=round(0.2 * SAMPLING_RATE))
            
#             corr_predictions, corr_labels = get_ith_labels_and_predictions(record_id=record_id, val_labels=val_labels, val_predictions=val_predictions, DS_EVAL=DS_EVAL)
            
#             # Post-processing enforce_min_spacing
#             corr_predictions = enforce_min_spacing(corr_predictions, min_distance=round(0.2 * SAMPLING_RATE))
#             # Plot prediction
#             check_ranges = [[0, 50000], [50000, 100000], [100000, 150000], [150000, 200000], [200000, 250000], [250000, 300000]]
#             for range in check_ranges:
#                 print("Plotting ECG for " + record_id, range)
#                 plot_ecg_predictions(filtered_waveform=filtered_waveform[range[0]:range[1],:], val_labels=padded_labels[range[0]:range[1]], val_predictions=corr_predictions[range[0]:range[1]], record_id=record_id, num_seconds=10)
#             # plot_ecg_predictions_random(filtered_waveform=filtered_waveform, val_labels=padded_labels, val_predictions=corr_predictions, record_id=record_id)
#             exit()

#             # # Get labels and prediction peaks
#             labels_peaks = sorted(get_peaks(ecg_signal=filtered_waveform, preferences=corr_labels))
#             predictions_peaks = sorted(get_peaks(ecg_signal=filtered_waveform, preferences=corr_predictions))
            

#             def rescale_peaks(peaks, original_fs, new_fs):
#                 """Rescales peak indices from new_fs back to original_fs."""
#                 return np.round(np.array(peaks) * (original_fs / new_fs)).astype(int)

#             # Rescale peaks to the original sampling rate
#             labels_peaks_rescaled = rescale_peaks(labels_peaks, fs, SAMPLING_RATE)
#             predictions_peaks_rescaled = rescale_peaks(predictions_peaks, fs, SAMPLING_RATE)

#             # Create annotation with rescaled peaks
#             total_symbol_rescaled = ['N' for _ in predictions_peaks_rescaled]
#             dir_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_predictions_atr")
#             os.makedirs(dir_test, exist_ok=True)
#             curr_dir = os.getcwd()
#             os.chdir(dir_test + '/')

#             # Save annotation
#             annotation_rescaled = wfdb.Annotation(
#                 record_name=record_id,
#                 extension='beat',
#                 sample=np.asarray(predictions_peaks_rescaled),  # Use rescaled peaks
#                 symbol=np.asarray(total_symbol_rescaled),
#                 fs=fs  # Set to the original sampling rate
#             )
#             annotation_rescaled.wrann(write_fs=True)

#             os.chdir(curr_dir)


def post_processing(val_data, val_predictions, val_labels, val_names, ecgpath="data/ecg_segmentation", DS_EVAL = DS_EVAL):

    val_labels = val_labels.reshape(len(DS_EVAL), -1)
    val_predictions = val_predictions.reshape(len(DS_EVAL), -1)     # (patent, min(len(waveform)))
    val_data=val_data.reshape(len(DS_EVAL), -1, 2) 

    dir_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_predictions_atr")
    os.makedirs(dir_test, exist_ok=True)
    for waveform, label, pred, name in zip(val_data, val_labels, val_predictions, np.unique(val_names)):
        # Plot prediction
        pred = enforce_min_spacing(pred, min_distance=100)
        # check_ranges = [[0, 50000], [50000, 100000], [100000, 150000], [150000, 200000], [200000, 250000], [250000, 300000]]
        # for range in check_ranges:
        #     print("Plotting ECG for ", name, range)
        #     plot_ecg_predictions(filtered_waveform=waveform[range[0]:range[1],:], val_labels=label[range[0]:range[1]], val_predictions=pred[range[0]:range[1]], record_id=name, num_seconds=10)
        
        
        # Extract peaks
        labels_peaks = sorted(get_peaks(ecg_signal=waveform, preferences=label))
        predictions_peaks = sorted(get_peaks(ecg_signal=waveform, preferences=pred))

        # Rescale peaks to original sampling rate
        labels_peaks_rescaled = rescale_peaks(labels_peaks, FS_ORG, SAMPLING_RATE)
        predictions_peaks_rescaled = rescale_peaks(predictions_peaks, FS_ORG, SAMPLING_RATE)

        print(labels_peaks_rescaled[:100], len(labels_peaks_rescaled))
        print(predictions_peaks_rescaled[:100], len(predictions_peaks_rescaled))


        total_symbol_rescaled = ['N' for _ in predictions_peaks_rescaled]

        annotation_filepath = os.path.join(dir_test, f"{name}.beat")
        curr_dir = os.getcwd()
        os.chdir(dir_test)
        print(f"Saving annotation for {name} at {annotation_filepath}")

        # Save annotation
        wfdb.wrann(
        record_name=name,
        extension="beat",
        sample=np.asarray(predictions_peaks_rescaled),
        symbol=np.asarray(total_symbol_rescaled),
        fs=FS_ORG 
        )
        os.chdir(curr_dir)
            
    
    
if __name__ == "__main__":
    all_val_data = np.load("data/ecg_segmentation/processed/val_data_subseq.npy")
    all_val_labels = np.load("data/ecg_segmentation/processed/val_labels_subseq.npy")
    all_val_names = np.load("data/ecg_segmentation/processed/val_names_subseq.npy")
    all_val_predictions = np.load("all_val_prediction.npy")
    
    
    # all_train_data = np.load("data/ecg_segmentation/processed/train_data_subseq.npy")
    # all_train_labels = np.load("data/ecg_segmentation/processed/train_labels_subseq.npy")
    # all_train_names = np.load("data/ecg_segmentation/processed/train_names_subseq.npy")
    # all_train_predictions = np.load("all_val_prediction.npy")
    # for data, label, pred, name in zip(all_val_data, all_val_labels, all_val_predictions, all_val_names):
    #     print(data.shape)
    #     print(label.shape)
    #     plot_ecg_predictions(filtered_waveform=data, val_labels=label, val_predictions=pred, record_id=name)
    # for i in range(all_val_labels.shape[0]):
    #     print("ACC", np.mean(all_val_labels[i, :] == all_val_predictions[i, :]))


    post_processing(val_data=all_val_data, val_predictions=all_val_predictions, val_labels=all_val_labels, val_names=all_val_names)
