import os, re
import glob # data import
import numpy as np # computing
import pandas as pd # datafames
from scipy.io import loadmat # data import
import matplotlib.pyplot as plt # plotting
import librosa # signal processing
import librosa.display # spectrograms
from scipy.signal import butter, sosfilt, sosfreqz # filtering


def butter_bandpass(lowcut, highcut, fs, order = 5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog = False, btype = "band", output = "sos")
        return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
        sos = butter_bandpass(lowcut, highcut, fs, order = order)
        y = sosfilt(sos, data)
        return y


def import_data(data_dir):
    
    # listing available .mat files
    data_files = glob.glob(data_dir + "S_**/*.mat", recursive = True)

    # sorting by subject's number
    data_files.sort()
    
    # removing weird participants
    removing_ppt = ["data/S_03/S_03.mat", "data/S_04/S_04.mat", "data/S_06/S_06.mat",
                    "data/S_09/S_09.mat", "data/S_12/S_12.mat", "data/S_25/S_25.mat"]
    data_files = [elem for elem in data_files if elem not in removing_ppt] 

    # importing all .mat data files into "data"
    data = [loadmat(f) for f in data_files]
    
    # printing the number of imported files
    print("Successfully imported data for", len(data_files), "participants")

    # returns the imported data
    return data, data_files


def trigger_formatting(com_emg, com_text, tick_rate, sample_rate_emg):
    """
    Trigger formatting
    For more info about .mat files exported by Labchart, see
    https://www.adinstruments.com/support/knowledge-base/how-does-matlab-open-exported-data

    Content of matrix com_emg
    - column 1: comchan (channel comment was made in, can be -1 for all channels
    - column 2: comblock (block comment was made in)
    - column 3: comtickpos (position of comment referring to tickrate of that block)
    - column 4: comtype (comment type 1=user, 2=event marker)
    - column 5: comtextmap (contains index that refers to column vector comtext)

    Converts comments from tickrate position to sample rate position
    the tick rate gives the maximum sample rate of all recorded channels
    which was 40kHz for the audio signal (discarded).
    """

    com_emg[:, 2] = com_emg[:, 2] / int(tick_rate / sample_rate_emg)
    com_emg[:, 2] = com_emg[:, 2].astype(int)

    # replaces comments' indices by comments' values
    for i in range(0, len(com_emg) ):
        com_emg[i, 4] = int(com_text[int(com_emg[i, 4]) - 1])
        
    # checks it there are duplicates
    print("Is there successive trigger values duplicates?", any(np.diff(com_emg[:, 4]) == 0) )

    # counts number of occurrences for each element
    babar = com_emg[:, 4].tolist()
    occurrences = {i:babar.count(i) for i in np.unique(babar)}
    print("Trigger occurrences before cleaning:", occurrences)

    # removes lines that contains zeros
    com_emg = com_emg[np.all(com_emg != 0, axis = 1)]
    # print(com_emg.shape)

    # retrieves triggers for the relaxation period
    com_relax = com_emg[0, :]

    # deletes this line
    com_emg = np.delete(arr = com_emg, obj = 0, axis = 0)
    # print(com_emg.shape)

    # removes extra lines that contains a 32 (erroneous relaxation trigger)
    com_emg = com_emg[np.all(com_emg != 32, axis = 1)]
    # print(com_emg.shape)

    babar = com_emg[:, 4].tolist()
    occurrences = {i:babar.count(i) for i in np.unique(babar)}
    
    # prints a warning if the length is not correct
    if len(com_emg) != 360:
        print("Error: length of the com matrix should be 360... looking for suspects")
        suspect = int([(k, v) for k, v in occurrences.items() if v == 61][0][0])
        suspect_line = np.where(com_emg[:, 4] == suspect)[0][-1]
        com_emg = np.delete(com_emg, (suspect_line), axis = 0)
        if len(com_emg) != 360:
            print("Error: length of the com matrix should be 360... looking for suspects again")
            suspect = int([(k, v) for k, v in occurrences.items() if v == 61][0][0])
            suspect_line = np.where(com_emg[:, 4] == suspect)[0][-1]
            com_emg = np.delete(com_emg, (suspect_line), axis = 0)
        else:
            print("Success!", com_emg.shape)
    
    # counts number of occurrences for each element
    babar = com_emg[:, 4].tolist()
    occurrences = {i:babar.count(i) for i in np.unique(babar)}
    print("Trigger occurrences after cleaning:", occurrences)
    
    return(com_emg, com_relax)
    

def import_trigger_values(com_emg, num_sujet):
    """
    Imports trigger IDs from the OpenSesame log file
    """
    # specifies path to file
    file = "data/" + num_sujet + "/" + num_sujet + ".csv"

    # reads the data in a dataframe
    df = pd.read_csv(file)

    # creates a sixth column for IDs in com_emg
    com_emg = np.column_stack([com_emg, df["id"]])
    
    return com_emg


def assembling_trigger(data_EMG, com_emg, com_relax, emg_sr, titles_emg):
    """
    Managing and assembling triggers
    """

    # number of item repetitions
    nb_repet = 6

    # relaxation period starts at the com_relax trigger and lasts for 60sec
    relax_time = 60
    relax_start = com_relax[2] / emg_sr
    relax_stop = relax_start + relax_time
    relax_data = data_EMG[int(com_relax[2]):int(com_relax[2] + relax_time * emg_sr), :]
    
    print("Shape of the resulting matrix for the relaxation period is:", relax_data.shape)

    # listen, overt and inner periods start at each of the other com triggers and they each last for 1s
    item_time = 1

    # extracts trigger values (which provide condition)
    # overt speech condition: trigger values = 1 to 20
    # inner speech condition: trigger values = 21 to 40
    # listening condition: trigger values = 41 to 60
    trigger_values = np.unique(com_emg[:, 5]) # , 'stable') # extract the 60 triggers IDs (in column 6 of "com")
    trigger = np.zeros(shape = (nb_repet * item_time * int(emg_sr), len(titles_emg), len(trigger_values) ) ) # initializes trigger matrix

    print("Shape of the resulting matrix for experimental trials is:", trigger.shape)
    
    return trigger, trigger_values


def emg_processing(data_EMG, trigger, com_emg, trigger_values, nb_repet, titles_emg, sample_rate_emg):
    """
    Process EMG data by channel (i.e., by muscle)
    """

    # duration of one trial (in seconds)
    item_time = 1
    
    # for each channel (i.e. muscle)
    for i in range(data_EMG.shape[1]):

        # removes mean EMG
        data_EMG[:, i] = data_EMG[:, i] - np.mean(data_EMG[:, i])

        # band-pass filtering from 20Hz to 450Hz with a fifth order filter
        data_EMG[:, i] = butter_bandpass_filter(data = data_EMG[:, i], lowcut = 20, highcut = 450, fs = sample_rate_emg, order = 5)

        # signal rectification
        # data_EMG[:, i] = abs(data_EMG[:, i])

    # saves time of all triggers associated with the same word
    times_word = np.zeros(shape = (len(trigger_values), nb_repet) )

    # concatenates nb_repet repetitions of each word
    for i in range(len(trigger_values) ): # makes a loop for each trigger value (60 in total)

        # find all rows corresponding to the trigger (same word in same condition)
        rows_trigger = np.where(com_emg[:, 5] == trigger_values[i])[0]
        
        # test
        # rows_trigger = np.where(com_emg[:, 5] == trigger_values[i])[0]; print(len(rows_trigger), rows_trigger); print(i); i+=1;

        # for each word, in each condition (6 repetitions of each word in total)
        for j in range(len(rows_trigger) ):

            # EMG data for each word in each condition
            trigger_temp = np.zeros(shape = (int(item_time * sample_rate_emg), len(titles_emg) ) )
            trigger_temp[:, :] = data_EMG[int(com_emg[rows_trigger[j], 2]):int(com_emg[rows_trigger[j], 2] + item_time * sample_rate_emg), :] 

            # start time of the trigger associated with the word
            times_word[i, j] = com_emg[rows_trigger[j], 2] / sample_rate_emg

            if "trigger_temp1" in locals():
                # if trigger_temp1 already exists, rowbinding six seconds of each word (6 repetitions)
                # trigger_temp1 = trigger_temp
                # trigger_temp1 = [trigger_temp1, trigger_temp]
                trigger_temp1 = np.vstack((trigger_temp1, trigger_temp) )
            else:
                # else, creating it
                trigger_temp1 = trigger_temp

            # removes trigger_temp
            del(trigger_temp)
            
            # checking shape during the j-loop
            # print(trigger_temp1.shape)

        trigger[:, :, i] = trigger_temp1
        del(trigger_temp1)
        
    return data_EMG, trigger, times_word


def plot_spectrogram(signal, sr, n_fft = 128, window = "hamming", win_length = 128, hop_length = 32):

    ########################################
    # plot 1 (raw EMG signal)
    ###################################
    
    plt.figure(figsize = [12, 8]) # defining the plotting area
    plt.subplot(2, 1, 1) # first plot
    
    plt.plot(signal)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.title("Raw EMG signal for some trial", loc = "left")
    
    ######################################################################
    # plot 2 (Short-term fourier transform spectrogram)
    #################################################################
    
    plt.subplot(2, 1, 2) # second plot
    
    # computes the STFT
    emg_stft = np.abs(
        librosa.stft(
            y = signal,
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            window = window
        )
    ) ** 2

    librosa.display.specshow(
        librosa.amplitude_to_db(
            emg_stft,
            ref = np.max
        ),
        y_axis = "linear",
        x_axis = "s",
        cmap = "viridis", # "RdBu_r",
        sr = sr,
        hop_length = hop_length
    )

    plt.title("STFT power spectrogram of the EMG signal for some trial", loc = "left")
    # plt.colorbar(format = "%+2.0f dB")
    plt.tight_layout()


#if __name__ == '__main__':
#    data_dir = 'data_sample'
#    df, data = parse_data(data_dir, verbose=False)
#    print(df)
#
#    idx, ch = 0, 0
#
#    sig, meta = data[idx], df.iloc[idx]
#    t = np.linspace(0, sig.shape[1]/meta.Fs, sig.shape[1])
#
#    fig, ax = plt.subplots()
#    ax.plot(t, sig[ch, :])
#    fig.show()
