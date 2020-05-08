from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt
import ripper
import sys

####################################################################
# FOR THE PROGRAM TO WORK CORRECTLY, USE THE FOLLOWING PARAMETERS: # 
#                                                                  #
#           FOR EXPERIMENT 38515 USE ROI (162000, 197700)          #
#           FOR EXPERIMENT 38516 USE ROI (175000, 230700)          #
#           FOR EXPERIMENT 38865 USE ROI (33000, 35000)            #
#           FOR EXPERIMENT 38882 USE ROI (130000, 155000)          #
#           FOR EXPERIMENT 38892 USE ROI (135000, 190000)          #
#                                                                  #
####################################################################

#
# DIRS WITH NEEDED FILES
#

stage = 0
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output", "sawtooth_frequency_computation_for_all_detectors")

pyglobus_dir = os.path.join(current_dir, "pyglobus", "python")

sys.path.append(pyglobus_dir)
try:
    import pyglobus
except ImportError as e:
    print("Cannot import pyglobus from %s, exiting" % pyglobus_dir)
    sys.exit(1)

#
# PARAMS FOR ALGORITMS:
#    DATA_FILE = NUMBER OF EXPERIMENT
#

DATA_FILE = 0
HIGH_PASS_CUTOFF = 250
LOW_PASS_CUTOFF = 2000
MOVING_AVERAGE_WINDOW_SIZE = 5

#
# NEEDED FUCTIONS 
#

# Plotting sample_data and saving it to PNG file
def plot(x, y, label_x, label_y, color="k", new_fig=True, flush=True):
    global stage

    if new_fig:
        plt.figure(figsize=(15, 10))


    global DATA_FILE
    
    plt.xlabel(label_x, fontsize=25)
    plt.ylabel(label_y, fontsize=25)
    
    if DATA_FILE != 38516:
        plt.plot(x[0], y[0], 'ro-')
    if DATA_FILE != 38515:
        plt.plot(x[1], y[1], 'bo-')
    plt.plot(x[2], y[2], 'ko-')
    plt.plot(x[3], y[3], 'go-')

    if DATA_FILE == 38516:
        plt.legend(("SXR 27 мкм","SXR 50 мкм","SXR 80 мкм"))
    elif DATA_FILE == 38515:
        plt.legend(("SXR 15 мкм","SXR 50 мкм","SXR 80 мкм"))
    else:
        plt.legend(("SXR 15 мкм","SXR 27 мкм","SXR 50 мкм","SXR 80 мкм"))

    if flush:
        out = os.path.join(output_dir, "#%i.png" % stage)
        plt.savefig(out)

        print("Stage %i result:" % stage, out)

        stage += 1


# Applies Butterworth filter
def butter_filter(input_, cutoff, fs, btype, order=5):
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype=btype, analog=False)
    return signal.filtfilt(b, a, input_)


# Applies moving average window
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# Enter the needed values to start algoritm
def init_data_():
    experiments_numbers = [38515,38516, 38865, 38882, 38892]

    print("Enter the number of experiment: \n")

    for i in range(len(experiments_numbers)):
        print(experiments_numbers[i])

    print("\n")
    
    global DATA_FILE
    DATA_FILE = int(input())

    c = False
    for i in range(len(experiments_numbers)):
        if DATA_FILE == experiments_numbers[i]:
            c = True
            break;

    if c != True:
        print("Bad input!")
        exit(1)

    os.system('cls')



if __name__ == "__main__":
    font = {"size": 22}

    plt.rc("font", **font)

    data_array_x = []
    data_array_y = []

    os.makedirs(output_dir, exist_ok=True)

    print("Stage %i: Data loading and preparing" % stage)

    arr = [18,19,20,26]

    init_data_()

    for j in range(len(arr)):

        data_all = ripper.extract('sample_data', DATA_FILE, [arr[j]])

        c,d = ripper.x_y(data_all[0][arr[j]])

        data = np.array([np.array(c),np.array(d)])

        print("Loaded %s" % DATA_FILE)

        roi = (175000, 230700)
        print(roi)
        x = data[0, roi[0]:roi[1]]
        y = data[1, roi[0]:roi[1]]

        sample_rate = 1.0 / (x[1] - x[0])

        y = butter_filter(y, HIGH_PASS_CUTOFF, sample_rate, btype="high")

        y = butter_filter(y, LOW_PASS_CUTOFF, sample_rate, btype="low")

        zero_crossings = np.where(np.diff(np.sign(y)))[0]

        print("Stage %i: Computing frequencies" % stage)

        freqs = []

        for i in range(len(zero_crossings) - 2):
            freqs.append(1 / (x[zero_crossings[i + 2]] - x[zero_crossings[i]]))

        x = x[zero_crossings][:-(MOVING_AVERAGE_WINDOW_SIZE + 1)]
        y = moving_average(freqs, MOVING_AVERAGE_WINDOW_SIZE)

        
        data_array_x.append(x)
        data_array_y.append(y)

    plot(data_array_x, data_array_y, "Время, с", "Частота, Гц", color="ko-")

    print("Done!")
