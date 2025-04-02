import numpy as np
import wfdb
import os
from downstream_clustering.clustering import plot_ecg_data
import neurokit2 as nk
from data.process.dataset_config import SAMPLING_RATE
from data.process.preprocessing import butter_bandpass_filter
from scipy.signal import resample
import random

def cal_rms(signal):
    return np.sqrt(np.mean(signal**2))

def add_noise_at_snr(original_signal, noise_signal, snr = -6):
    
    rms_org = cal_rms(original_signal)
    rms_noise = cal_rms(noise_signal)
    alpha = rms_org / (rms_noise * 10 ** (snr / 20))

    signal_added_noise = original_signal + alpha * noise_signal

    return signal_added_noise

def save_signal(signal, save_path = "./data/ecg_clustering/mit-bih-noise-stress-test-database-1.0.0/", name = "", fs = 250, channels = 2):
    record_name = os.path.join(save_path, name)
    # Tạo đối tượng Record
    record = wfdb.Record(
        record_name=record_name,
        n_sig=channels,
        fs=fs,
        sig_len=len(signal),
        sig_name=['ECG'],
        p_signal=np.expand_dims(signal, axis=1)
    )
    wfdb.wrsamp(record_name=record_name, fs=fs, units=['mV'], sig_name=['ECG'], p_signal=record.p_signal) # save .dat & .hea file

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

def preprocess_ECG(ecgpath="data/ecg_segmentation", processedecgpath="data/ecg_clustering/processed", reprocess=False, change_resolution = False):
    
    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    # record_ids = RECORD_IDS
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    remove_record_ids = ['102', '104', '107', '217', '118', '119']
    record_ids = [x for x in record_ids if x not in remove_record_ids]
    all_ecgs = []
    all_noise_signal = np.load("data/ecg_clustering/processed/all_ecgs_subseq.npy", mmap_mode='r')[2160:]
    count = 0
    # Loop through records to create ecgs and labels
    for record_id in sorted(record_ids):
        count += 1
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
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

        all_ecgs.append(filtered_waveform[:,:].T)

        # Normalize denoise
        if count == 10:
            
            signal_lens = [sig.shape[1] for sig in all_ecgs]
            all_ecgs = np.array([sig[:,:min(signal_lens)] for sig in all_ecgs]) # lay lai cung mot do dai nho nhat

            # Normalize ecgs and changes it to be batch, time, channel
            all_ecgs = denoiseECG(all_ecgs) 
            
            print("Denoise and Normalize completed!")
            os.makedirs(processedecgpath, exist_ok=True)
            print("Begin create subsequences...")
            T = all_ecgs.shape[1]                                                         # bc its been transposed
            subseq_size= 2500

            all_ecgs_subseq = np.stack(np.split(all_ecgs[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
            all_ecgs_subseq = np.reshape(all_ecgs_subseq, (-1, all_ecgs_subseq.shape[2], all_ecgs_subseq.shape[3]))
            
            all_ecgs_subseq_noisy = [x if ((idx // 18) % 2 == 0) else add_noise_at_snr(x, all_noise_signal[random.randint(0, 539)], snr = 0) for idx, x in enumerate(all_ecgs_subseq)]
            all_ecgs_subseq_noisy = np.array(all_ecgs_subseq_noisy)

            np.save(os.path.join(processedecgpath, f"{record_id}.npy"), all_ecgs_subseq)
            print(f"========= SAVE DATASET {record_id}DONE ===========")
            np.save(os.path.join(processedecgpath, f"{record_id}e00.npy"), all_ecgs_subseq_noisy)
            print(f"========= SAVE DATASET {record_id}e00 DONE ===========")
            
            all_ecgs = []
            count = 0
            all_ecgs_subseq_noisy = []

    print("========= SAVE DATASET 00 DONE ===========")
preprocess_ECG()
# all_subseq = np.load("data/ecg_clustering/processed/113e00.npy")
# noise_signal = add_noise_at_snr(all_subseq[0], all_subseq[-1], snr=00)
# print(all_subseq.shape)
# plot_ecg_data(all_subseq[580])
