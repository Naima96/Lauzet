# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:46:56 2022

@author: al-abiad
"""
import numpy as np
from scipy import signal
from scipy import stats
import pandas as pd

def calc_skew(a):

    mean = np.mean(a)
    m2 = _moment(a, 2,mean=mean)
    m3 = _moment(a, 3,mean=mean)

    vals = m3 / m2**1.5
    skew=vals

    return skew


def calc_kurt(a):

    mean = np.mean(a)
    m2 = _moment(a, 2,mean=mean)
    m4 = _moment(a, 4, mean=mean)
    valsk = m4 / m2**2.0
    kurt=valsk - 3 
    return kurt


def calc_frequency_params(signal_x, sampling_rate=100):

    sampling_rate=100
    nfft=1024
    nfft2=512
    cutoff=12.0

    signal_x = signal_x-np.mean(signal_x)
    dim = signal_x.shape
    
    freq = (np.fft.fftfreq(nfft) * sampling_rate)[0:nfft2]
    haming= np.hamming(dim[0])
    sp_hat = np.fft.fft(signal_x*haming, nfft)
    furval = sp_hat[0:nfft2] * np.conjugate(sp_hat[0:nfft2])

    idx1 = freq <= cutoff
    idx_cutoff = np.argwhere(idx1)
    #all freq less than cutoff
    freq = freq[idx_cutoff]
    #keep values less than cutoff
    sp = furval[idx_cutoff]
    #normalise
    sp_norm = sp / sum(sp)


    max_freq = freq[sp_norm.argmax()][0] # Feature 1: dominant freq 
    max_freq_val = sp_norm.max().real    #Feature 2: magnitude of dominant freq
    idx2 = (freq > max_freq - 0.5) * (freq < max_freq + 0.5)  
    idx_freq_range = np.where(idx2)[0]
    dom_freq_ratio = sp_norm[idx_freq_range].real.sum() #Feature 3: dominant frequency ratio

    # Calculate Feature 4: spectral flatness 
    spectral_flatness = 10.0*np.log10(stats.mstats.gmean(sp_norm)/np.mean(sp_norm))

    # Calculate Feature 5: Estimate spectral entropy
    spectral_entropy_estimate = 0
    for isess in range(len(sp_norm)):
        if sp_norm[isess] != 0:
            logps = np.log2(sp_norm[isess])
        else:
            logps = 0
        spectral_entropy_estimate = spectral_entropy_estimate - logps * sp_norm[isess]

    spectral_entropy_estimate = spectral_entropy_estimate / np.log2(len(sp_norm))


    return max_freq,max_freq_val,dom_freq_ratio,spectral_flatness[0].real,spectral_entropy_estimate[0].real





def findMiddle(input_list):
    middle = float(len(input_list))/2
    return np.where(middle % 2 != 0,int(middle - .5),int(middle))



def calc_index_min(data):
   return np.argmax(data)


def calc_sma(data):
   return sum(list(map(abs, data)))



def calc_median(mag):
    return np.median(mag)

def calc_rms(data):
    return np.sqrt(np.mean(data ** 2))

def calc_mean_abs_deviation(data):
    return stats.median_abs_deviation(data)

def calc_energy(data):
    squares = data ** 2
    return squares.sum()

def _moment(a, moment, mean=None):

    
    if moment==2:
        n_list=np.array([2])
    elif moment==3:
       n_list=np.array([3,1])
       
    elif moment==4:
       n_list=np.array([4,2]) 
    

    a_zero_mean = a - mean
    
    if n_list[-1] == 1:
        s = a_zero_mean.copy()
    else:
        s = a_zero_mean**2

    # Perform multiplications
    for n in n_list[-2::-1]:
        s = s**2
        if n % 2:
            s *= a_zero_mean
    return np.mean(s)

def calc_MCR(data):
    data=data-np.mean(data)
    MCR=0
    for ki in range(len(data) - 1):
        if np.sign(data[ki]) != np.sign(data[ki + 1]):
            MCR += 1
    return MCR

def _calc_feature_window(signal_data,window_size,overlap):
    
    window_size=int(window_size*100)
    overlap=int(overlap*100)
    
    features=[]
    for i in range(0,len(signal_data)-window_size,overlap):
        
        median=calc_median(signal_data[i,i+window_size])
        skew=calc_skew(signal_data[i,i+window_size])
        kurt=calc_kurt(signal_data[i,i+window_size])
        max_freq,max_freq_val,dom_freq_ratio,spectral_flatness,spectral_entropy_estimate=calc_frequency_params(signal_data[i,i+window_size])
        SMA=calc_sma(signal_data[i,i+window_size])
        energy=calc_energy(signal_data[i,i+window_size])
        mad=calc_mean_abs_deviation(signal_data[i,i+window_size])
        rms=calc_rms(signal_data[i,i+window_size])
        std=np.std(signal_data[i,i+window_size])
        minimum=np.amin(signal_data[i,i+window_size])
        maximum=np.amax(signal_data[i,i+window_size])
        MCR=calc_MCR(signal_data[i,i+window_size])
        
        features.append([median,std,skew,kurt,SMA,energy,mad,rms,minimum,maximum,MCR,max_freq,max_freq_val,dom_freq_ratio,spectral_flatness,spectral_entropy_estimate])
        
    
    features_df=pd.DataFrame([features])
    return(features_df)
        
        
    
    
    