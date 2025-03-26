from data.process.dataset_config import NUM_SAMPLES_PER_FRAME, OFFSET_SAMPLE
import collections
from collections import Counter
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


def change_resolution_labels_with_peaks(data, label, NUM_SAMPLES_PER_FRAME):
    new_labels = []
    
    for seq_idx, subseq in enumerate(label):
        calib_labels = []
        for i in range(0, len(subseq) - NUM_SAMPLES_PER_FRAME + 1, NUM_SAMPLES_PER_FRAME):
            most_common_label = int(Counter(subseq[i : i + NUM_SAMPLES_PER_FRAME]).most_common(1)[0][0])
            calib_labels.append(most_common_label)

        new_calib_labels = calib_labels[:]  # Copy to modify
        i = 0
        while i < len(calib_labels):
            if calib_labels[i] == 1:
                group_indices = [i]
                
                j = i + 1
                while j < len(calib_labels) and calib_labels[j] == 1:
                    group_indices.append(j)
                    j += 1
                
                if len(group_indices) > 1:
                    max_value = -np.inf
                    best_index = group_indices[0]
                    
                    for idx in group_indices:
                        start = idx * NUM_SAMPLES_PER_FRAME
                        end = min(start + NUM_SAMPLES_PER_FRAME, len(data[seq_idx]))
                        block_data = data[seq_idx][start:end]
                        
                        max_block_value = np.max(np.abs(block_data))
                        
                        # Compear current peak of current block with previous best block
                        if max_block_value > max_value:
                            max_value = max_block_value
                            best_index = idx
                    
                    # Set Only the Best Block to '1' other will be zero
                    for idx in group_indices:
                        new_calib_labels[idx] = 1 if idx == best_index else 0
                
                i = j  # Skip processed blocks
            else:
                i += 1
        new_labels.append(new_calib_labels)
    
    return np.array(new_labels)


def change_resolution_labels(tensor, NUM_SAMPLES_PER_FRAME):
    new_tensor = []
    for subseq in tensor:
        calib_labels = []
        for i in range(0, len(subseq) - NUM_SAMPLES_PER_FRAME + 1, NUM_SAMPLES_PER_FRAME):  
            most_common_label = int(Counter(subseq[i : i + NUM_SAMPLES_PER_FRAME]).most_common(1)[0][0])
            calib_labels.append(most_common_label)
        new_tensor.append(calib_labels)
    return np.array(new_tensor)


def butter_bandpass(lowcut=4, highcut=50, fs=250, order=2):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut=4, highcut=50, fs=250, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def is_beat(annotation):
    beat_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'E']
    return 1 if annotation in beat_labels else 0


def enforce_min_spacing(padded_labels, min_distance=100):
    """Ensure consecutive 1s in padded_labels are at least min_distance apart using middle points of 1s."""
    
    # Find all consecutive regions of 1s
    beat_regions = []
    in_beat = False
    
    for i in range(len(padded_labels)):
        if padded_labels[i] == 1 and not in_beat:
            start = i
            in_beat = True
        elif padded_labels[i] == 0 and in_beat:
            end = i - 1
            in_beat = False
            middle = (start + end) // 2  # Middle point of the 1s region
            beat_regions.append(middle)

    if in_beat:  # Handle case where the last segment extends to the end
        middle = (start + len(padded_labels) - 1) // 2
        beat_regions.append(middle)

    # Keep only regions that are min_distance apart
    filtered_indices = [beat_regions[0]]
    for i in range(1, len(beat_regions)):
        if beat_regions[i] - filtered_indices[-1] >= min_distance:
            filtered_indices.append(beat_regions[i])

    # Reset padded_labels and mark only valid beats with sample_offset
    new_labels = np.zeros_like(padded_labels)
    for index in filtered_indices:
        if index < OFFSET_SAMPLE:
            new_labels[:index + OFFSET_SAMPLE] = 1
        elif index >= len(padded_labels) - OFFSET_SAMPLE:
            new_labels[index - OFFSET_SAMPLE:] = 1
        else:
            new_labels[index - OFFSET_SAMPLE: index + OFFSET_SAMPLE] = 1

    return new_labels



def plot_waveform(waveform, filtered_waveform, fs=250, title="ECG Signal Before and After Filtering"):
    """
    Plot the original and filtered ECG waveforms.
    
    Parameters:
    """
    time = np.arange(waveform.shape[0]) / fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, waveform[:, 0], label="Original", alpha=0.7)
    plt.plot(time, filtered_waveform[:, 0], label="Filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()