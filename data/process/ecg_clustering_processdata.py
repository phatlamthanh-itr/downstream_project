import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import wfdb
import neurokit2 as nk
from data.process.dataset_config import SAMPLING_RATE
from wfdb.processing import resample_multichan
from data.process.preprocessing import is_beat, butter_bandpass_filter, change_resolution_labels, change_resolution_labels_with_peaks
from scipy.signal import resample

def main():
    # the repo designed to have files live in /rebar/data/ecg/
    downloadextract_ECGfiles()
    preprocess_ECGdata(change_resolution=False)

def downloadextract_ECGfiles(zippath="data/ecg.zip", targetpath="data/ecg_clustering", redownload=False):
    if os.path.exists(targetpath) and redownload == False:
        print("ECG files already exist")
        return

    link = "https://www.physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip"
    print("Downloading ECG files (700 MB) ...")
    download_file(link, zippath)

    print("Unzipping ECG files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_ECGdata(ecgpath="data/ecg_clustering", processedecgpath="data/ecg_clustering/processed", reprocess=False, change_resolution = False):
    # if os.path.exists(processedecgpath) and reprocess == False:
    #     print("ECG data has already been processed")
    #     return
    
    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-noise-stress-test-database-1.0.0")) if '.dat' in file]

    all_ecgs = []
    all_names = []

    # Loop through records to create ecgs and labels
    for record_id in sorted(record_ids):
        # Import recording and annotations
        record_path = os.path.join(ecgpath,"mit-bih-noise-stress-test-database-1.0.0", record_id)
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']

        # Resample waveform
        fs = record.fs 
        if fs != SAMPLING_RATE:
            num_samples = int(waveform.shape[0] * (SAMPLING_RATE / fs))  
            waveform = resample(waveform, num_samples)

        # Apply filters
        filtered_waveform = butter_bandpass_filter(waveform)
        
        # Normalize
        filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
        filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)

        all_ecgs.append(filtered_waveform[:,:].T) #shape: 650000x2 -> 2x650000
        all_names.append(record_id)
 
    all_names = np.array(all_names)
    signal_lens = [sig.shape[1] for sig in all_ecgs]
    all_ecgs = np.array([sig[:,:min(signal_lens)] for sig in all_ecgs]) # lay lai cung mot do dai nho nhat

    # Normalize ecgs and changes it to be batch, time, channel
    all_ecgs = denoiseECG(all_ecgs) # shape: (15, 650000, 2)
    print("Denoise and Normalize completed!")

    # # Save ecgs to file
    os.makedirs(processedecgpath, exist_ok=True)
    np.save(os.path.join(processedecgpath, "all_ecgs.npy"), all_ecgs)
    np.save(os.path.join(processedecgpath, "all_names.npy"), all_names)

# #+++++++++++++++++++++++++++++++++TO DO HERE+++++++++++++++++++++++++++ 

    # # Process subseq dataset
    print("Begin create subsequences...")
    T = all_ecgs.shape[1]                                                         # bc its been transposed
    subseq_size= 2500
    all_ecgs_subseq = np.stack(np.split(all_ecgs[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    all_ecgs_subseq = np.reshape(all_ecgs_subseq, (-1, all_ecgs_subseq.shape[2], all_ecgs_subseq.shape[3]))
    all_names_subseq = np.repeat(all_names, (T // subseq_size))
    print(all_ecgs_subseq.shape)
    np.save(os.path.join(processedecgpath, "all_names_subseq.npy"), all_names_subseq)
    np.save(os.path.join(processedecgpath, "all_ecgs_subseq.npy"), all_ecgs_subseq)
    print("========= SAVE DATASET DONE ===========")


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