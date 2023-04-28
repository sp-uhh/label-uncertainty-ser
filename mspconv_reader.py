from scipy.signal import butter,filtfilt
from torch import index_select, tensor
from scipy.io import wavfile
from enum import Enum
import pandas as pd
import numpy as np
import librosa
import fnmatch
import torch
import os

# Annotation Filtering constants
window_size = 0.5 #mins
default_num_annot = 6

# Dataset constants
batch_size = 25
sample_size = 12 # secs
segment_size = 300 # (sample_size*batch_size) # secs

# Required Sample rate of annotation/labels
samplerate_annot = 25 #25 #59 #fps

# Audio Sample rate
samplerate_audio = 16000 #22050 #Hz

# Annotation Cleaning constants
to_filter_annot = ["001", "007", "009"]
filter_annot = True

# Dataset Location Paths
labels_folder = 'INSERT Path OF Annotations/ folder'
partitions_file = 'INSERT Path OF partitions.txt'
parts_info_loc = 'INSERT Path OF  Time_Labels/conversation_parts.txt'
audios_folder = 'INSERT Path of Audios/ folder'

class Labels(Enum):
    arousal = "Arousal"
    valence = "Valence"
    dominance = "Dominance"
default_label = Labels.arousal.value

class Partition(Enum):
    train = "Train"
    test = "Test"
    dev = "Validation"
default_partition = Partition.train.value

# 1. Luz Martinez-Lucas, Mohammed Abdelwahab, and Carlos Busso, "The MSP-conversation corpus," in Interspeech 2020, Shanghai, China, October 2020, pp. 1823-1827.

# NOTE:
#       Main function: 
#           read_mspconv(partition)
#               * reads the labels and audios for a particular partition
#               * performs filtering and pre-processing of labels
#               * performs shaping of audio and respective labels for training.
#           Sample call : 
#               trainset = Partition().train
#               audio, all_label, mean_label, std_label, time = read_mspconv(trainset)  


# MAIN FUNCITON TO READ DATASET - For a Partition
def read_mspconv(partition):
    files = get_files_for_partition(partition)
    # Init data holders
    audio_data = None
    all_lbl_data = None
    mean_lbl_data = None
    std_lbl_data = None
    time_data = None
    for file in files: # ["MSP-Conversation_0079"]
        parts = get_parts_for_file(file)
        for part in parts:
            part_audio = read_audio_for_part(file, part)
            part_lbl_all, part_lbl_mean, part_lbl_std, time = get_part_labels(file, part)
            # Cut extra segment secs, wrt selected sample size
            part_audio, total_duration = cut_batch_audio_for_eqduration(part_audio)
            part_lbl_all, part_lbl_mean, part_lbl_std, time = cut_batch_labels_for_eqduration(part_lbl_all, part_lbl_mean, part_lbl_std, time, total_duration)

            if audio_data is None:
                audio_data = part_audio
                all_lbl_data, mean_lbl_data, std_lbl_data = part_lbl_all, part_lbl_mean, part_lbl_std 
                time_data = time 
            else:
                audio_data = np.concatenate((audio_data, part_audio), 0)
                all_lbl_data = np.concatenate((all_lbl_data, part_lbl_all), 0)
                mean_lbl_data = np.concatenate((mean_lbl_data, part_lbl_mean), 0)
                std_lbl_data = np.concatenate((std_lbl_data, part_lbl_std), 0)
                time_data = np.concatenate((time_data, time), 0)
                
    return torch.from_numpy(audio_data), torch.from_numpy(all_lbl_data), torch.from_numpy(mean_lbl_data), torch.from_numpy(std_lbl_data), torch.from_numpy(time_data)



## Audio readers
def read_audio_for_part(filename, part, samplerate=samplerate_audio):
    start_time, end_time = get_start_end_time(filename, part)
    data = read_audio_part(get_audio_loc(filename), start_time, end_time, samplerate)
    return data

def read_audio_part(filename, start, end, samplerate=None):
    if samplerate:
        data, srate = librosa.load(filename, sr=samplerate, offset=start, duration=end-start)
    else:
        data, srate = librosa.load(filename, offset=start, duration=end-start)
    return data

def get_audio_loc(filename):
    loc = audios_folder + filename + ".wav"
    return loc

## Annotations Label Readers
def get_part_labels(file, part):
    print("*"*75, " For AROUSAL ", "*"*75)
    part_aro_all, time = read_all_labels_for_part(file, part, Labels.arousal.value, default_num_annot)
    part_aro_mean, part_aro_std = read_mean_std_labels_for_part(part_aro_all)
    print("*"*75, " For VALENCE ", "*"*75)
    part_val_all, time = read_all_labels_for_part(file, part, Labels.valence.value, default_num_annot)
    part_val_mean, part_val_std = read_mean_std_labels_for_part(part_val_all)
    
    part_lbl_all  = form_label_struct(part_aro_all, part_val_all)
    part_lbl_mean = form_label_struct(part_aro_mean, part_val_mean)
    part_lbl_std  = form_label_struct(part_aro_std, part_val_std)

    return part_lbl_all, part_lbl_mean, part_lbl_std, time


def read_all_labels_for_part(filename, part, emo_dim, num_annotations):
    annot_filenames = get_annotations_files_for_part(emo_dim, filename, part)
    annot_folder = get_annotations_folder(emo_dim)

    start_time, end_time = get_start_end_time(filename, part)
    part_annotations = None
    for file in annot_filenames:
        annotations, time  = read_annotation_csv(annot_folder+file, starttime=start_time, endtime=end_time)
        # Low pass filter annotations, only faulty once.
        annotations = butter_lowpass_filter(annotations, file) if file.split("_")[-1].split(".")[0] in to_filter_annot and filter_annot else annotations
        if part_annotations is None:
            part_annotations = np.expand_dims(annotations, 1)
        else:
            part_annotations = np.column_stack((part_annotations, annotations)) 
    return part_annotations, time




# Label filtering and pre-processing
def get_annotations_in_timerange(start, end, timeframe, dataframe, prev):
    func = lambda d: d >= start and d < end
    filter_fn = np.vectorize(func)(timeframe)
    window_index = np.where(filter_fn)
    window_annotations = dataframe[window_index]
    if len(window_annotations) == 0:
        window_annotations = np.array([0.0, 0.0]) + prev
    return window_annotations

def fix_samplerate_medianfilter(data, timeframe, duration, samplerate):
    # timeframe, endtime, samplerate in frame per seconds fps
    # start = 0.00 # in secs
    window_center = round(1/samplerate, 2) # window_size/2
    half_wind = window_size/2
    step = round(1/samplerate, 2)
    print("Fixing Samplerate @ ", samplerate, " fps, step: ", step, " secs, half_wind: ", half_wind, " fs, window_center: ", window_center, " fs.")

    sampled_data = []
    sampled_time = []
    while(window_center <= duration):
        win_start = round(window_center - half_wind, 2) if window_center - half_wind >=0.0 else 0.0
        win_end   = round(window_center + half_wind, 2) if window_center + half_wind <=duration else duration

        prev = sampled_data[-1] if len(sampled_data) != 0 else 0.0
        sampled_data.append(np.median(get_annotations_in_timerange(win_start, win_end, timeframe, data, prev)))
        sampled_time.append(window_center)
        window_center   = round(window_center + step, 2)
    return np.array(sampled_data), np.array(sampled_time)

def butter_lowpass_filter(data, filename):
    print("Low pass filtering file: ", filename)
    # Filter requirements
    fs = 25.0       # sample rate, Hz
    cutoff = 0.25 #(Tuning of  cutoff frequency required), desired cutoff frequency of the filter, Hz , slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 3       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    print("Filter at Cutoff - ", normal_cutoff)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y




# Utilities
def get_files_for_partition(partition="Train"):
    files = np.loadtxt(partitions_file, delimiter=";", dtype='str')
    return files[files[:,1] == partition][:,0]

def read_annotation_csv(filepath, starttime, endtime, filter=True):
    labels_pd = pd.read_csv(filepath, header=None).to_numpy()
    labels_np = labels_pd[9: , :].astype(np.float) # Removing Metadata 
    if filter:
        # TODO: Check replace - https://numpy.org/doc/stable/reference/generated/numpy.ma.median.html
        resampled_annot, resampled_time = fix_samplerate_medianfilter(labels_np[:,1], labels_np[:,0], endtime-starttime, samplerate_annot)
        return resampled_annot, resampled_time
    else:
        return np.array(labels_np[:,1]), np.array(labels_np[:,0])

def read_mean_std_labels_for_part(all_part_annotations):
    mean_annotations = np.mean(all_part_annotations, axis=-1)
    std_annotations = np.std(all_part_annotations, axis=-1)
    return mean_annotations, std_annotations

def form_label_struct(aro_data, val_data, dom_data=None):
    aro_data = np.expand_dims(aro_data, axis=len(aro_data.shape))
    val_data = np.expand_dims(val_data, axis=len(val_data.shape))
    lbl_data = np.concatenate((aro_data, val_data), axis=-1)
    return lbl_data

def get_annotations_files_for_part(emo_dim, filename, part):
    folder = get_annotations_folder(emo_dim)
    annot_filenames = []
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, get_partname(filename, part)+'_*.csv'):
            annot_filenames.append(file)
    return np.array(annot_filenames)

def get_part_index(filename):
    part_index = filename.split("_")[-1]
    return part_index

def get_parts_for_file(filename):
    parts = np.loadtxt(parts_info_loc, delimiter=";", dtype='str')
    file_parts = parts[np.where(np.char.find(parts[:,0], filename)>=0)]
    vsplitter = np.vectorize(get_part_index)
    file_parts = vsplitter(file_parts[:,0])
    return file_parts

def get_annotations_folder(emo_dim=default_label):
    annotations_folder = labels_folder + emo_dim + "/"
    return annotations_folder

def get_partname(filename, part):
    partname = filename+ "_" + part 
    return partname

def get_annotations_loc(emo_dim, filename, part, annotator):
    annotations_loc = get_annotations_folder(emo_dim) + get_partname(filename, part) + "_" + annotator + ".csv"
    return annotations_loc

def get_file_startend(file_parts):
    first_part, last_part = file_parts[0,:], file_parts[-1,:]
    return float(first_part[1]), float(last_part[2])

def adjust_fileparts_with_file_start(file_parts, file_start):
    float_start = file_parts[:, 1].astype(np.float)
    float_end = file_parts[:, 2].astype(np.float)

    file_parts[:, 1] = float_start - file_start
    file_parts[:, 2] = float_end - file_start
    return file_parts

def get_start_end_time(filename, part):
    parts = np.loadtxt(parts_info_loc, delimiter=";", dtype='str')
    file_parts = parts[np.where(np.char.find(parts[:,0], filename)>=0)]
    file_start, file_end = get_file_startend(file_parts)
    file_parts = adjust_fileparts_with_file_start(file_parts, file_start)
    part_info = file_parts[np.where(np.char.find(file_parts[:,0], filename+"_"+part)>=0)][0]
    return float(part_info[1]), float(part_info[2])




# UTILITIES for Shaping Audio and Labels
def cut_batch_audio_for_eqduration(part_audio, samplesize=sample_size):
    audio_points = len(part_audio)
    audio_secs   = audio_points/samplerate_audio

    # Required Datapoints
    num_batches =  int(audio_secs/samplesize)
    num_datapoints = samplerate_audio*samplesize*num_batches
    # Ignore samples after duration. e.g. For audio of 186.9 secs, ignore 6.9 secs if sample_size*batch_size=180.  
    part_audio = part_audio[:num_datapoints]

    print("Num batches for audio - ", audio_secs, " secs, num of batches = ", num_batches)
    # Reshape as batch 
    part_audio = np.reshape(part_audio, (num_batches, int(len(part_audio)/num_batches)))
    return part_audio, int(samplesize*num_batches)

def cut_batch_labels_for_eqduration(part_lbl_all, part_lbl_mean, part_lbl_std, time, duration):
    assert len(part_lbl_all) == len(part_lbl_all) == len(part_lbl_all) == len(part_lbl_all), "ERROR in Cutting LABELS!!!!!!!!!!!!!, Length mismatch between labels." 
    # Duration in secs - samplesize*num_batches e.g. For audio 300 sec & samplesize = 12, num_batches = 25 and duration = 12*25 = 300 secs
    num_datapoints = int(duration*samplerate_annot) 
    num_batches = int(duration/sample_size)
    # Required Datapoints
    # Ignore samples after num_datapoints (samplesize*num_batches)
    part_lbl_all = part_lbl_all[:num_datapoints]
    part_lbl_mean = part_lbl_mean[:num_datapoints]
    part_lbl_std = part_lbl_std[:num_datapoints]
    time = time[:num_datapoints]

    # Reshape as batch 
    len_data, num_annot, num_emodims = part_lbl_all.shape
    part_lbl_all = np.reshape(part_lbl_all, (num_batches, int(len_data/num_batches), num_annot, num_emodims))
    part_lbl_mean = np.reshape(part_lbl_mean, (num_batches, int(len_data/num_batches), num_emodims))
    part_lbl_std = np.reshape(part_lbl_std, (num_batches, int(len_data/num_batches), num_emodims))
    time = np.reshape(time, (num_batches, int(len(time)/num_batches)))

    return part_lbl_all, part_lbl_mean, part_lbl_std, time