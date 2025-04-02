import os
import torch
import numpy as np
import wfdb
import matplotlib.pyplot as plt 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import subprocess
from downstream.downstream_config import downstream_config, CombinedBCEDiceLoss, DiceLoss
from utils.utils import printlog
from downstream.downstream_nets import LSTMDecoder, TransformerDecoder, LSTMDecoder2500
from downstream.trainer import train_model
from data.process.dataset_config import SAMPLING_RATE, DS_EVAL, NUM_SAMPLES_PER_FRAME, FS_ORG
from data.process.preprocessing import enforce_min_spacing
from downstream.utils import plot_ecg_predictions, plot_ecg_predictions_random, get_peaks, rescale_peaks, reconstruct, filter_peaks, filter_large_ones




def eval_beat_segmentation(downstream_model, train_data, train_labels, val_data, val_labels):
    """
    Perform downstream Beat Segmentation tasks
    """
 
    # --------- Set up dataloader for downstream training -------------------
    train_loader = downstream_model.setup_dataloader(data=train_data, label=train_labels, batch_size = downstream_config.batch_size ,train=True)
    val_loader = downstream_model.setup_dataloader(data=val_data, label=val_labels, batch_size = downstream_config.batch_size, train=False)
    
    # ------- Train downstream model -----------------------------------------
    best_loss, best_acc = train_model(model=downstream_model, train_loader=train_loader, train_labels=train_labels, val_loader=val_loader, 
                val_labels=val_labels)

    return  best_loss, best_acc



def post_processing(label_size, pred_path = downstream_config.save_pred_dir, save_beat_dir = downstream_config.save_beat_dir, ecgpath="data/ecg_segmentation", DS_EVAL = DS_EVAL, visualize = False):
    """
    Compare model prediction and labels, then save to .beat files
    """
    
    # Load pre-saved validation labels
    try:
        val_data = np.load(f"{ecgpath}/processed/val_data_subseq.npy")                      # Recover validation subseq waveform
        val_labels = np.load(f"{ecgpath}/processed/val_labels_subseq.npy")                  # Recover validation subseq labels
        val_names =  np.load(f"{ecgpath}/processed/val_names_subseq.npy")                   
        val_predictions = np.load(f"{pred_path}/val_predictions.npy")               
    except:
        raise Exception("Validation labels not found")

    # Change to (n_patients, min(len(waveform)), 2)
    val_data = val_data.reshape(len(DS_EVAL), -1, 2)
    val_labels = val_labels.reshape(len(DS_EVAL), -1)       # Change to (n_patents, min_len(waveform))
    if label_size == 2500:
        val_predictions = val_predictions.reshape(len(DS_EVAL), -1)     # (patent, min(len(waveform)))
    else:
        val_predictions = reconstruct(val_predictions=val_predictions)

    for waveform, label, pred, name in zip(val_data, val_labels, val_predictions, np.unique(val_names)):
        # if name not in ['231','222','105']: continue
        # pred = enforce_min_spacing(pred, min_distance=85)   
        # # if visualize:
        # check_ranges = [[0, 50000], [50000, 100000], [100000, 150000], [150000, 200000], [200000, 250000]]
        # for range in check_ranges:
        #     print("Plotting ECG for ", name, range)
        #     predict_range = np.zeros_like(label[range[0]:range[1]])
        #     # predictions_peaks = filter_peaks(sorted(get_peaks(ecg_signal=waveform, preferences=pred[range[0]:range[1]])))
        #     # predict_range[predictions_peaks] = 1
        #     plot_ecg_predictions_random(filtered_waveform=waveform[range[0]:range[1], :], val_labels=label[range[0]:range[1]], val_predictions=pred[range[0]:range[1]], record_id=name, num_seconds=10)

        # Extract peaks
        labels_peaks = sorted(get_peaks(ecg_signal=waveform, preferences=label))

        # Prediction post-processing
    
        predictions_filtered = filter_large_ones(pred, min_size=15)     # Filter out 1's noise 
        predictions_peaks = sorted(get_peaks(ecg_signal=waveform, preferences=predictions_filtered))  # Get peaks positions
        predictions_peaks = filter_peaks(predictions_peaks, min_distance=50)            # Ensure no two peaks are too close 


        # Rescale peaks to original sampling rate  (250 --> 360Hz) for creating annotation
        labels_peaks_rescaled = rescale_peaks(labels_peaks, FS_ORG, SAMPLING_RATE)
        predictions_peaks_rescaled = rescale_peaks(predictions_peaks, FS_ORG, SAMPLING_RATE)

        total_symbol_rescaled = ['N' for _ in predictions_peaks_rescaled]
        annotation_filepath = os.path.join(save_beat_dir, f"{name}.beat")
        curr_dir = os.getcwd()
        os.chdir(save_beat_dir)
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

    print("---------Post processing done!----------")
    

    # Run script to generate EC57 result
    subprocess.run(["./gen_ec57.sh", downstream_config.save_beat_dir, ecgpath + "/mit-bih-arrhythmia-database-1.0.0"], check=True, capture_output=True, text=True)

    print("---------EC57 result generated!----------")








