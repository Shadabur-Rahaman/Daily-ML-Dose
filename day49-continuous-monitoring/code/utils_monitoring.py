import numpy as np

def moving_average(arr, window=10):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def alert_if_threshold(value, threshold, name):
    if value > threshold:
        print(f"⚠️ {name} exceeded threshold ({value:.2f} > {threshold})")
