import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from data.process.dataset_config import SAMPLING_RATE, DS_TRAIN, DS_EVAL, OFFSET_SAMPLE, NUM_SAMPLES_PER_FRAME
from wfdb.processing import resample_multichan
from data.process.preprocessing import is_beat, butter_bandpass_filter, change_resolution_labels, change_resolution_labels_with_peaks

def main():
    # the repo designed to have files live in /rebar/data/ecg/
    downloadextract_ECGfiles()
    preprocess_ECGdata(change_resolution=False)


def downloadextract_ECGfiles(zippath="data/ecg.zip", targetpath="data/ecg_segmentation", redownload=False):
    # if os.path.exists(targetpath) and redownload == False:
    #     print("ECG files already exist")
    #     return

    link = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    print("Downloading ECG files (440 MB) ...")
    download_file(link, zippath)

    print("Unzipping ECG files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_ECGdata(ecgpath="data/ecg_segmentation", processedecgpath="data/ecg_segmentation/processed", reprocess=False, change_resolution = False):
    # if os.path.exists(processedecgpath) and reprocess == False:
    #     print("ECG data has already been processed")
    #     return
    
    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    
    all_ecgs = []
    all_labels = []
    all_names = []
    # Loop through records to create ecgs and labels
    for record_id in sorted(record_ids):
        
        # Import recording and annotations
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']
        annotation = wfdb.rdann(record_path, 'atr')
        sample = annotation.sample
        labels = annotation.symbol
        fs = record.fs
        # Resample to 250Hz
        if fs != SAMPLING_RATE:
            waveform, resampled_ann = resample_multichan(waveform, annotation, fs, SAMPLING_RATE)
            labels = resampled_ann.symbol
            sample = resampled_ann.sample
        # Apply filters
        filtered_waveform = butter_bandpass_filter(waveform)
        
        # Normalize
        filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
        filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
        
        # Create labels
        padded_labels = np.zeros(len(waveform))
        for i,l in enumerate(labels):
            if i==len(labels)-1:
                padded_labels[sample[i]:] = is_beat(l)
            else:
                padded_labels[sample[i] - OFFSET_SAMPLE // 2: sample[i] + OFFSET_SAMPLE // 2] = is_beat(l)

        # padded_labels = padded_labels[sample[0]:]
        # Post-processing (make sure min_distance between two peaks greater than 0.4 s)
        # padded_labels = enforce_min_spacing(padded_labels=padded_labels, min_distance=round(0.4 * SAMPLING_RATE))
        
        # Plot ECG (optional)
        # check_ranges = [[0, 50000], [50000, 100000], [100000, 150000], [150000, 200000], [200000, 250000], [250000, 300000]]
        # for range in check_ranges:
        #     print("Plotting ECG for " + record_id, range)
        #     plot_ecg_predictions(filtered_waveform=filtered_waveform[range[0]:range[1],:], val_labels=padded_labels[range[0]:range[1]], val_predictions=padded_labels[range[0]:range[1]], record_id=record_id)
        
        all_labels.append(padded_labels)
        all_ecgs.append(filtered_waveform[:,:].T)
        all_names.append(record_id)

    all_names = np.array(all_names)
    signal_lens = [sig.shape[1] for sig in all_ecgs]
    # label_lens = [label.shape[0] for label in all_labels]
    all_ecgs = np.array([sig[:,:min(signal_lens)] for sig in all_ecgs])
    all_labels = np.array([label[:min(signal_lens)] for label in all_labels])
    
    train_data = np.array([all_ecgs[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN])
    train_labels = np.array([all_labels[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN])
    train_names = np.array([[record_id for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN]])


    val_data = np.array([all_ecgs[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL])
    val_labels = np.array([all_labels[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL])    
    val_names = np.array([[record_id for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL]])


    # Normalize ecgs aand changes it to be batch,time, channel
    train_data, val_data = denoiseECG(train_data), denoiseECG(val_data)
    print("Denoise and Normalize completed!")

    # # Save ecgs to file
    os.makedirs(processedecgpath, exist_ok=True)

    np.save(os.path.join(processedecgpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedecgpath, "train_labels.npy"), train_labels)
    np.save(os.path.join(processedecgpath, "train_names.npy"), train_names)

    np.save(os.path.join(processedecgpath, "val_data.npy"), val_data)
    np.save(os.path.join(processedecgpath, "val_labels.npy"), val_labels)
    np.save(os.path.join(processedecgpath, "val_names.npy"), val_names)

# #+++++++++++++++++++++++++++++++++TO DO HERE+++++++++++++++++++++++++++ 

    # # Process subseq dataset
    T = train_data.shape[1]                                                         # bc its been transposed
    subseq_size= 2500
    train_data = np.stack(np.split(train_data[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    train_data = np.reshape(train_data, (-1, train_data.shape[2], train_data.shape[3]))
    
    train_labels = np.stack(np.split(train_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    train_labels = np.reshape(train_labels, (-1, train_labels.shape[2]))
    train_names = np.repeat(train_names, (T // subseq_size))


    # Split into 0.125s chunks
    if change_resolution:
        # train_labels = change_resolution_labels(train_labels, NUM_SAMPLES_PER_FRAME)
        train_labels = change_resolution_labels_with_peaks(train_data, train_labels, NUM_SAMPLES_PER_FRAME)

    label_sq_size = train_labels.shape[1]

    np.save(os.path.join(processedecgpath, 'train_data_subseq.npy'), train_data)
    np.save(os.path.join(processedecgpath, f'train_labels_subseq_{label_sq_size}.npy'), train_labels)
    np.save(os.path.join(processedecgpath, "train_names_subseq.npy"), train_names)

    T = val_data.shape[1]
    val_data = np.stack(np.split(val_data[:, :subseq_size* (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    val_data = np.reshape(val_data, (-1, val_data.shape[2], val_data.shape[3]))
    val_labels = np.stack(np.split(val_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    val_labels = np.reshape(val_labels, (-1, val_labels.shape[2]))
    val_names = np.repeat(val_names, (T // subseq_size))


    # val_labels = change_resolution_labels_with_peaks(val_data,val_labels, NUM_SAMPLES_PER_FRAME=NUM_SAMPLES_PER_FRAME)
    if change_resolution:
        # val_labels = change_resolution_labels(val_labels, NUM_SAMPLES_PER_FRAME)
        val_labels = change_resolution_labels_with_peaks(val_data, val_labels, NUM_SAMPLES_PER_FRAME=NUM_SAMPLES_PER_FRAME)
    label_sq_size = val_labels.shape[1]

    np.save(os.path.join(processedecgpath, "val_data_subseq.npy"), val_data)
    np.save(os.path.join(processedecgpath, f'val_labels_subseq_{label_sq_size}.npy'), val_labels)
    np.save(os.path.join(processedecgpath, "val_names_subseq.npy"), val_names)


    print("========= SAVE DATASET DONE ===========")


def plot_ecg_predictions(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels, marking peaks in predicted beats.
 
    """
    num_samples = num_seconds * sample_rate
    time_axis = np.linspace(0, num_seconds, num_samples)

    # Get the first channel only
    ecg_signal = filtered_waveform[:num_samples, 0]
    predictions = val_predictions[:num_samples]
    labels = val_labels[:num_samples]

    labels_peaks = get_peaks(ecg_signal, labels)
    predictions_peaks = get_peaks(ecg_signal, predictions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot ECG waveform
    axs[0].plot(time_axis, ecg_signal, color='black', label="Filtered ECG")
    axs[0].scatter(np.array(labels_peaks) / sample_rate, ecg_signal[labels_peaks], color='red', marker='o', label="Label Peaks")
    axs[0].scatter(np.array(predictions_peaks) / sample_rate, ecg_signal[predictions_peaks], color='green', marker='x', label="Predicted Peaks")
    axs[0].set_ylabel("ECG Signal")
    axs[0].legend()

    # Plot Ground Truth Labels
    axs[1].plot(time_axis, labels, color='blue', linestyle='dashed', label="True Labels")
    axs[1].scatter(np.array(labels_peaks) / sample_rate, np.ones(len(labels_peaks)), color='red', marker='o', label="Label Peaks")
    axs[1].set_ylabel("True Labels (0/1)")
    axs[1].legend()

    # Plot Model Predictions
    axs[2].plot(time_axis, predictions, color='red', linestyle='dotted', label="Predictions")
    axs[2].scatter(np.array(predictions_peaks) / sample_rate, np.ones(len(predictions_peaks)), color='green', marker='x', label="Predicted Peaks")
    axs[2].set_ylabel("Predictions (0/1)")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].legend()

    plt.suptitle(f"ECG Signal, Labels, and Predictions for Record {record_id}", fontsize=16)

    plt.tight_layout()
    plt.show()

def get_peaks(ecg_signal, preferences):
    """
    Get the position of peak or prediction or labels
    """
    peaks = []
    in_peak = False
    for i in range(len(preferences)):
        if preferences[i] == 1:
            if not in_peak:
                start = i  # Start of a detected beat
                in_peak = True
            # Find max within detected 1s
            if i == len(preferences) - 1 or preferences[i + 1] == 0:
                peak_idx = start + np.argmax(ecg_signal[start:i+1])     # Peak index --> With max absolute value
                peaks.append(peak_idx)
                in_peak = False

    return peaks


def denoiseECG(data, hz=250):
    data_filtered = np.empty(data.shape)
    for n in range(data_filtered.shape[0]):
        for c in range(data.shape[1]):
            newecg = nk.ecg_clean(data[n,c,:], sampling_rate=hz)
            data_filtered[n,c] = newecg
    data = data_filtered

    feature_means = np.mean(data, axis=(2))
    feature_std = np.std(data, axis=(2))
    data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

    data = np.transpose(data, (0,2,1))

    return data

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename


if __name__ == "__main__":
    main()