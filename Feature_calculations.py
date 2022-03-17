# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:46:56 2022

@author: al-abiad
"""
import numpy as np
from scipy import signal


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

def dominant_frequency(signal_x): #100
    sampling_rate=100
    nfft=1024
    nfft2=512
    fmin=0.5
    fmax=4

    signal_x = signal_x-np.mean(signal_x)
    dim = signal_x.shape
    
    freq = (np.fft.fftfreq(nfft) * sampling_rate)[0:nfft2]
     
    lowind=np.where(freq>fmin)[0][0]
    upind=np.max(np.where(freq<fmax))

    haming= np.hamming(dim[0])
    sp_hat = np.fft.fft(signal_x*haming, nfft)
    
    furval = sp_hat[0:nfft2] * np.conjugate(sp_hat[0:nfft2])

    
    ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
    domfreq=freq[ind] 
    
    return domfreq



def findMiddle(input_list):
    middle = float(len(input_list))/2
    return np.where(middle % 2 != 0,int(middle - .5),int(middle))



def calc_index_min(data):
   return np.argmax(data)


def calc_sma(data):
   return sum(list(map(abs, data)))



def calc_median(mag):
    return np.median(mag)


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

def _calc_feature_window(signal_data):
    
    
    