import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt 
from data.process.dataset_config import SAMPLING_RATE, DS_EVAL, NUM_SAMPLES_PER_FRAME, FS_ORG



# ======================== PLOT FUNCTIONS =========================================

def plot_ecg_predictions(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels.
 
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
    plt.suptitle("ECG Signal Before and After Filtering with record ID: " + str(record_id))
    plt.tight_layout()
    plt.show()


def plot_ecg_predictions_random(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels
    """
    total_samples = min(len(filtered_waveform), len(val_predictions), len(val_labels))
    num_samples = num_seconds * sample_rate

   
    if total_samples <= num_samples:
        start_idx = 0  
    else:
        start_idx = np.random.randint(0, total_samples - num_samples)

    end_idx = start_idx + num_samples
    time_axis = np.linspace(start_idx / sample_rate, end_idx / sample_rate, num_samples)

    # Extract the first channel 
    ecg_signal = filtered_waveform[start_idx:end_idx, 0]
    predictions = val_predictions[start_idx:end_idx]
    labels = val_labels[start_idx:end_idx]


    labels_peaks = get_peaks(ecg_signal, labels)
    predictions_peaks = get_peaks(ecg_signal, predictions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot ECG waveform
    axs[0].plot(time_axis, ecg_signal, color='black', label="Filtered ECG")
    axs[0].scatter(np.array(labels_peaks) / sample_rate + start_idx / sample_rate, ecg_signal[labels_peaks], color='red', marker='o', label="Label Peaks")
    axs[0].scatter(np.array(predictions_peaks) / sample_rate + start_idx / sample_rate, ecg_signal[predictions_peaks], color='green', marker='x', label="Predicted Peaks")
    axs[0].set_ylabel("ECG Signal")
    axs[0].legend()

    # Plot Ground Truth Labels
    axs[1].plot(time_axis, labels, color='blue', label="True Labels")
    axs[1].scatter(np.array(labels_peaks) / sample_rate + start_idx / sample_rate, np.ones(len(labels_peaks)), color='red', marker='o', label="Label Peaks")
    axs[1].set_ylabel("True Labels (0/1)")
    axs[1].legend()

    # Plot Model Predictions
    axs[2].plot(time_axis, predictions, color='brown', label="Predictions")
    axs[2].set_ylabel("Predictions (0/1)")
    axs[2].scatter(np.array(predictions_peaks) / sample_rate + start_idx / sample_rate, np.ones(len(predictions_peaks)), color='green', marker='x', label="Predicted Peaks")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].legend()
    
    fig.suptitle(f"ECG Signal, Ground Truth, and Predictions (Record ID: {record_id})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()






# ============================= HELPER FUNCTIONS ===============================================
def reconstruct(val_predictions):
    """
    Reconstruct labels and prediction to (patents, min(waveform))
    """
    # Reshape to (patent, 14400)
    all_val_prediction = val_predictions.reshape(len(DS_EVAL), -1)

    reconstructed_labels = []
    reconstructed_predictions = []

    # Reconstruct
    for i in range(all_val_prediction.shape[0]):
        predictions_per_patent = []
        for j in range(all_val_prediction.shape[1]):
            predictions_per_patent.extend([all_val_prediction[i][j] for _ in range(NUM_SAMPLES_PER_FRAME)])
        reconstructed_predictions.append(predictions_per_patent)
    reconstructed_predictions = np.array(reconstructed_predictions)

    return reconstructed_predictions


def get_ith_labels_and_predictions(record_id, val_labels, val_predictions, DS_EVAL):
    """
    Get i-th labels and predictions
    """
    search_index = DS_EVAL.index(int(record_id))
    print("Find labels at", search_index, record_id)
    return val_predictions[search_index, :], val_labels[search_index, :]



def rescale_peaks(peaks, original_fs, new_fs):
    """Rescales peak indices from new_fs back to original_fs."""
    return np.round(np.array(peaks) * (original_fs / new_fs)).astype(int)


def get_peaks(ecg_signal, preferences):
    """
    Get the position of peak or prediction or labels
    Args:
    - ecg_signal: 1D array of ecg signal
    - preferences: 1D array (list) of labels or predictions
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
                peak_idx = start + np.argmax(np.abs(ecg_signal[start:i+1]))     # Peak index --> With max absolute value
                peaks.append(peak_idx)
                in_peak = False

    return peaks