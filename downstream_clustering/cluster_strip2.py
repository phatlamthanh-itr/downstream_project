import os
import torch
import numpy as np
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
import argparse
from utils.utils import printlog, load_data, import_model, init_dl_program
from tqdm import tqdm
import wfdb
import neurokit2 as nk
from data.process.dataset_config import SAMPLING_RATE
from scipy.signal import resample
from data.process.preprocessing import butter_bandpass_filter
import joblib
from multiprocessing import Process, Queue, cpu_count, Pool, Manager, set_start_method


all_expconfigs = {**allrebar_expconfigs}

def process_ECG_error_from_strip2(rebar_model, kmean_model, path="data/strip2", processedecgpath="data/ecg_clustering/processed"):
    print("Processing ECG data from strip2...")

    queue = Queue()
    writer = Process(target=writer_process, args=(queue, "data/ecg_clustering/noise.txt", "data/ecg_clustering/clean.txt"))
    writer.start()
    with open("./data/ecg_clustering/errror.txt", 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines):
        record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(path, line)) if '.dat' in file]
        record_path = os.path.join(path, os.path.join(line, f"{record_ids[0]}"))

        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']

        # Resample waveform
        fs = record.fs 
        if fs != SAMPLING_RATE:
            num_samples = int(waveform.shape[0] * (SAMPLING_RATE / fs))  
            waveform = resample(waveform, num_samples)

        waveform, flag_get_sample = get_sample_from_hea(record_path, waveform)
        if not flag_get_sample:
            continue

        # Apply filters
        filtered_waveform = butter_bandpass_filter(waveform)
        
        # Normalize
        filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
        filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
        nan_count = np.isnan(filtered_waveform).sum()
        if filtered_waveform.shape[1] == 3:
            if nan_count == 0:
                try: 
                    denoised_waveform = denoiseECG(np.expand_dims(filtered_waveform[:, [0, 1]].T, axis=0))
                    encoded_waveform = rebar_model.encode(denoised_waveform)
                    encoded_waveform = np.max(encoded_waveform, axis=1)
                    label = kmean_model.predict(encoded_waveform)
                    queue.put((label, line))
                except:
                    with open('data/ecg_clustering/error_filted.txt', 'a') as f:
                        f.write(f"{line}\n")

            else:
                with open('data/ecg_clustering/error_nan.txt', 'a') as f:
                    f.write(f"{line}\n")
        else:
            with open('data/ecg_clustering/error_channel.txt', 'a') as f:
                f.write(f"{line}\n")
    queue.put(None)
    writer.join()

def process_ECG_from_strip2(rebar_model, kmean_model, path="data/strip2", processedecgpath="data/ecg_clustering/processed"):
    print("Processing ECG data from strip2...")

    queue = Queue()
    writer = Process(target=writer_process, args=(queue, "data/ecg_clustering/noise.txt", "data/ecg_clustering/clean.txt"))
    writer.start()
    count = 0
    all_ecgs = []
    all_names = []
    for study_ID in tqdm(sorted(os.listdir(path)), desc= "Process ECG in strips 2: "):
        for record_name in sorted(os.listdir(os.path.join(path, study_ID))):
            try:
                record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(path, os.path.join(study_ID, record_name))) if '.dat' in file]
                record_path = os.path.join(path, os.path.join(study_ID, os.path.join(record_name, f"{record_ids[0]}")))

                record = wfdb.rdrecord(record_path)
                waveform = record.__dict__['p_signal']

                # Resample waveform
                fs = record.fs 
                if fs != SAMPLING_RATE:
                    num_samples = int(waveform.shape[0] * (SAMPLING_RATE / fs))  
                    waveform = resample(waveform, num_samples)

                waveform, flag_get_sample = get_sample_from_hea(record_path, waveform)
                if not flag_get_sample:
                    continue

                # Apply filters
                filtered_waveform = butter_bandpass_filter(waveform)
                
                # Normalize
                filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
                filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
                nan_count = np.isnan(filtered_waveform).sum()
                if filtered_waveform.shape[1] == 3:
                    if nan_count == 0:
                        all_ecgs.append(filtered_waveform[:,[0, 1]].T) # shape: (15000, 3) -> (3, 15000)
                        all_names.append(f"{study_ID}/{record_name}")
                        count += 1
                    else:
                        with open('data/ecg_clustering/error_nan.txt', 'a') as f:
                            f.write(f"{study_ID}/{record_name}\n")
                else:
                        with open('data/ecg_clustering/error_channel.txt', 'a') as f:
                            f.write(f"{study_ID}/{record_name}\n")
            except:
                continue

            if count >= 3000:
                all_names = np.array(all_names)
                min_signal_lens = min([sig.shape[1] for sig in all_ecgs])
                all_ecgs = np.array([sig[:,:min_signal_lens] for sig in tqdm(all_ecgs)]) # lay lai cung mot do dai nho nhat
                # all_ecgs = np.array(all_ecgs)
                # Normalize ecgs and changes it to be batch, time, channel
                all_ecgs = denoiseECG(all_ecgs) # shape
                if all_ecgs.shape != (3000, 2500, 2):
                    with open('data/ecg_clustering/errror.txt', 'a') as f:
                        f.write("\n".join(all_names) + "\n")
                else:
                    encoded_all_ecg = rebar_model.encode(all_ecgs)
                    encoded_all_ecg = np.max(encoded_all_ecg, axis=1)
                    labels = kmean_model.predict(encoded_all_ecg)
                    # denoised_waveform = denoiseECG(np.expand_dims(filtered_waveform[:, [0, 1]].T, axis=0))
                    # encoded_waveform = rebar_model.encode(denoised_waveform)
                    # encoded_waveform = np.max(encoded_waveform, axis=1)
                    # label = kmean_model.predict(encoded_waveform)
                    queue.put((labels, all_names))
                count = 0
                all_ecgs = []
                all_names = []

    queue.put(None)
    writer.join()

    # process_single_ecg(study_ID, record_name, path, rebar_model=rebar_model, kmean_model=kmean_model)
def writer_process(queue, noise_file, clean_file, batch_size=1):
    buffer_noise = []
    buffer_clean = []
    
    while True:
        item = queue.get()
        if item is None:
            break

        labels, lines = item
        if int(labels) == 0:
            buffer_noise.append(lines)
        else:
            buffer_clean.append(lines)
        
        # for idx, label in enumerate(labels):
        #     if label == 0:
        #         buffer_noise.append(lines[idx])
        #     else:
        #         buffer_clean.append(lines[idx])

        if len(buffer_noise) >= batch_size:
            with open(noise_file, 'a') as f:
                f.write("\n".join(buffer_noise) + "\n")
            buffer_noise = []

        if len(buffer_clean) >= batch_size:
            with open(clean_file, 'a') as f:
                f.write("\n".join(buffer_clean) + "\n")
            buffer_clean = []

    if buffer_noise:
        with open(noise_file, 'a') as f:
            f.write("\n".join(buffer_noise) + "\n")

    if buffer_clean:
        with open(clean_file, 'a') as f:
            f.write("\n".join(buffer_clean) + "\n")

def process_single_ecg(queue, study_ID, record_name, path, rebar_model, kmean_model):
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(path, os.path.join(study_ID, record_name))) if '.dat' in file]
    record_path = os.path.join(path, os.path.join(study_ID, os.path.join(record_name, f"{record_ids[0]}")))

    record = wfdb.rdrecord(record_path)
    waveform = record.__dict__['p_signal']

    # Resample waveform
    fs = record.fs 
    if fs != SAMPLING_RATE:
        num_samples = int(waveform.shape[0] * (SAMPLING_RATE / fs))  
        waveform = resample(waveform, num_samples)

    waveform, flag_get_sample = get_sample_from_hea(record_path, waveform)
    if not flag_get_sample:
        return

    # Apply filters
    filtered_waveform = butter_bandpass_filter(waveform)

    # Normalize
    filtered_waveform -= np.mean(filtered_waveform, axis=0)
    filtered_waveform /= np.std(filtered_waveform, axis=0)

    if filtered_waveform.shape[1] == 3 and not np.isnan(filtered_waveform).sum():

        denoised_waveform = denoiseECG(np.expand_dims(filtered_waveform[:, [0, 1]].T, axis=0))
        encoded_waveform = rebar_model.encode(denoised_waveform)
        encoded_waveform = np.max(encoded_waveform, axis=1)
        label = kmean_model.predict(encoded_waveform)

        queue.put((label, f"{study_ID}/{record_name}"))

        # if int(label) == 0:
        #     print(f"{study_ID}/{record_name}")
        #     save_name("./data/ecg_clustering/noise.txt", f"{study_ID}/{record_name}")
        #     # plot_ecg_data(filtered_waveform[:, [0, 1]])
        # else:
        #     save_name("./data/ecg_clustering/clean.txt", f"{study_ID}/{record_name}")



def save_name(path_to_save, name):
    with open(path_to_save, 'a') as noise_file:
        noise_file.write(name + '\n')

# def denoiseECG(data, hz=250):
#     data_filtered = np.empty(data.shape)
#     for n in range(data_filtered.shape[0]):
#         for c in range(data.shape[1]):
#             newecg = nk.ecg_clean(data[n,c,:], sampling_rate=hz)
#             data_filtered[n,c] = newecg
#     data = data_filtered

#     feature_means = np.mean(data, axis=(2))
#     feature_std = np.std(data, axis=(2))
#     data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

#     data = np.transpose(data, (0,2,1))

#     return data

def denoiseECG(data, hz=250, min_length = 20):
    data_filtered = np.empty(data.shape)
    for n in range(data_filtered.shape[0]):
        for c in range(data.shape[1]):
            segment = data[n, c, :]
            if len(segment) < min_length:
                pad_width = min_length - len(segment)
                segment = np.pad(segment, (0, pad_width), mode='constant')
            newecg = nk.ecg_clean(segment, sampling_rate=hz)
            data_filtered[n, c] = newecg[:data.shape[2]]
    data = data_filtered

    feature_means = np.mean(data, axis=(2))
    feature_std = np.std(data, axis=(2))
    data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

    data = np.transpose(data, (0,2,1))
    return data

def get_sample_from_hea(record_path, waveform):
    header_path = record_path + '.hea'
    startSample = 0
    stopSample = 0
    with open(header_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'startSample:' in line:
                try:
                    startSample = int(line.split(':')[1].strip())
                except Exception as e:
                    print("Error parse startSample:", e)
            elif 'stopSample:' in line:
                try:
                    stopSample = int(line.split(':')[1].strip())
                except Exception as e:
                    print("Error parse stopSample:", e)
    if (stopSample - startSample) < 2500 or len(waveform) - startSample < 2500:
        return waveform, False
    else:
        return waveform[startSample: startSample + 2500], True    

GLOBAL_REBAR_MODEL = None
GLOBAL_KMEAN_MODEL = None

def init_worker(config):
    """
    Hàm initializer cho pool: thiết lập các biến global cho worker.
    Lưu ý: nếu model chứa các CUDA tensors, hãy chuyển về CPU nếu cần.
    """
    global GLOBAL_REBAR_MODEL, GLOBAL_KMEAN_MODEL
    GLOBAL_REBAR_MODEL = import_model(config, reload_ckpt = True)
    GLOBAL_KMEAN_MODEL = joblib.load('./experiments/out/cluster/kmeans/clustering_pipeline_1.pkl')

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
    if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", config.data_name, config.run_dir, "checkpoint_best.pkl"))):
        train_data, _, val_data, _, _, _ = load_data(config = config, data_type = "fullts")
        model = import_model(config, train_data=train_data, val_data=val_data)
        model.fit()
    
    config.set_inputdims(2)
    rebar_model = import_model(config, reload_ckpt = True)
    # rebar_model = rebar_model.cpu()
    if os.path.exists('./experiments/out/cluster/kmeans/clustering_pipeline_118.pkl'):
        pipeline = joblib.load('./experiments/out/cluster/kmeans/clustering_pipeline_118.pkl')
        print("Load KMeans model success!")
    process_ECG_error_from_strip2(rebar_model=rebar_model, kmean_model=pipeline)
    exit()
    process_ECG_from_strip2(rebar_model=rebar_model, kmean_model=pipeline)