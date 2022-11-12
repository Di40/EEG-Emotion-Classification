#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:54:13 2020

@author: dichoski
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import signal



filename='s01.dat'
with open(filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
data=subject['data']
for i in range (32):
    with open(filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
    data=subject['data']
    data=data[30,31, 384:]
    
    sf = 128.
    time = np.arange(data.size) / sf
    
    # # Plot the signal
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # plt.plot(time, data, lw=1.5, color='k')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Voltage')
    # plt.xlim([time.min(), time.max()])
    # plt.title('EEG data')
    # sns.despine()
    
    
    
    
    # Define window length (4 seconds)
    win = 4 * sf
    freqs, psd = signal.welch(data, sf, nperseg=win)
    
    # # Plot the power spectrum
    # sns.set(font_scale=1.2, style='white')
    # plt.figure(figsize=(8, 4))
    # plt.plot(freqs, psd, color='k', lw=2)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (V^2 / Hz)')
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    # plt.xlim([0, freqs.max()])
    # sns.despine()
    
    
    # Define delta lower and upper limits
    band = [4,8,12,30,45]
    
    # Find intersecting values in frequency vector
    idx_theta = np.logical_and(freqs >= band[0], freqs <= band[1])
    idx_alpha = np.logical_and(freqs >= band[1], freqs <= band[2])
    idx_beta = np.logical_and(freqs >= band[2], freqs <= band[3])
    idx_gamma = np.logical_and(freqs >= band[3], freqs <= band[4])
    # Plot the power spectral density and fill the delta area
    plt.figure(figsize=(7, 5))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_theta, color='blue')
    plt.fill_between(freqs, psd, where=idx_alpha, color='red')
    plt.fill_between(freqs, psd, where=idx_beta,  color='green')
    plt.fill_between(freqs, psd, where=idx_gamma, color='purple')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([4, 45])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    sns.despine()