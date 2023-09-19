import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks, detrend, butter, filtfilt


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_heart_rate(peaks, fs):
    # Calculate time difference between peaks
    time_diff = np.diff(peaks) / fs
    # Convert time difference to heart rate
    heart_rates = 60 / time_diff
    return np.mean(heart_rates)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

