# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:27:32 2022

@author: al-abiad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

def read_data(path):
    fs=100
    
    filename=os.path.join(path,"ACC.txt" )
    df=pd.read_csv(filename, delimiter=",")
    time_acc=df["Time [s]"].values
    time_start_acc=time_acc[0]
    acce_uncalib=df.iloc[:,1:4]
    acce_calib=df.iloc[:,4:7]
    acce_uncalib.index=time_acc
    acce_calib.index=time_acc
    
    filename=os.path.join(path,"GYRO.txt" )
    df=pd.read_csv(filename, delimiter=",")
    time_gyro=df["Time(s)"].values
    time_start_gyro=time_gyro[0]
    gyro_uncalib=df.iloc[:,1:4]
    gyro_calib=df.iloc[:,4:7]
    
    gyro_uncalib.index=time_gyro
    gyro_calib.index=time_gyro
    
    time_start=np.minimum(time_start_gyro,time_start_acc)
    
    acce_calib = acce_calib[~acce_calib.index.duplicated()]
    acce_uncalib = acce_uncalib[~acce_uncalib.index.duplicated()]

    gyro_calib = gyro_calib[~gyro_calib.index.duplicated()]
    gyro_uncalib = gyro_uncalib[~gyro_uncalib.index.duplicated()]
    
    
    acce_calib=acce_calib[acce_calib.index>time_start]
    acce_uncalib=acce_uncalib[acce_uncalib.index>time_start]
    
    gyro_calib=gyro_calib[gyro_calib.index>time_start]
    gyro_uncalib=gyro_uncalib[gyro_uncalib.index>time_start]
    
    acce_calib.index=acce_calib.index-acce_calib.index[0]
    
    acce_uncalib.index=acce_uncalib.index-acce_uncalib.index[0]
    
    gyro_calib.index=gyro_calib.index-gyro_calib.index[0]
    gyro_uncalib.index=gyro_uncalib.index-gyro_uncalib.index[0]

    matrix1=acce_calib
    matrix2=gyro_calib

    t_t=np.linspace(0, matrix1.index[len(matrix1)-1], num=np.int((matrix1.index[len(matrix1)-1])*fs), endpoint=True,dtype=np.float32)
    matrix1=matrix1.reindex(matrix1.index.union(t_t))
    matrix2=matrix2.reindex(matrix2.index.union(t_t))

    matrix1=matrix1.interpolate(method='linear', limit_direction='both', axis=0)
    matrix2=matrix2.interpolate(method='linear', limit_direction='both', axis=0)

    matrix1=matrix1[matrix1.index.isin(pd.Index(t_t))]
    matrix2=matrix2[matrix2.index.isin(pd.Index(t_t))]

    matrix1.index=np.around(matrix1.index.values.astype('float64'),decimals=4)
    matrix2.index=np.around(matrix2.index.values.astype('float64'),decimals=4)

    acc_interp=matrix1
    gyro_interp=matrix2
    
    return acc_interp,gyro_interp

def plot_data(dataframe):
    columns=dataframe.columns
    fig, ax=plt.subplots(3,figsize=(25,15))
    
    ax[0].plot(dataframe[columns[0]])
    ax[1].plot(dataframe[columns[1]])
    ax[2].plot(dataframe[columns[2]])

    font = {'family' : 'cursive',
            'style':'normal',
            'weight': 2,
            'size'   : 20}
    plt.rc('font', **font)
    
    plt.subplots_adjust(hspace=0.5)

    ax[0].set_ylabel(columns[0])
    ax[1].set_ylabel(columns[1])
    ax[2].set_ylabel(columns[2])
    
    ax[2].set_ylabel('samples')
    ax[0].set_title('signal')
    
def filter_data(dataframe,cut_off,order):
    fc=cut_off
    fs=100
    N=order
    Wn =fc/(fs/2) # Cutoff frequency normalized 
    B, A = signal.butter(N, Wn,'low', output='ba') 
    dataframe_filtered=dataframe.copy()
    dataframe_filtered=signal.filtfilt(B, A, dataframe_filtered)
    
    return dataframe_filtered

def calculate_norm(dataframe):
    matrix1=dataframe
    x=matrix1.iloc[:,0].values**2
    y=matrix1.iloc[:,1].values**2
    z=matrix1.iloc[:,2].values**2
    m=x+y+z
#    m=np.sqrt(m)
    mm=np.array([np.sqrt(i) for i in m])
    
    return mm
    

def plot_norm(norm,signal_type='acc'):
    fig, ax=plt.subplots(1,figsize=(25,15))
    
    ax[0].plot(norm)

    font = {'family' : 'cursive',
            'style':'normal',
            'weight': 2,
            'size'   : 20}
    plt.rc('font', **font)

    if signal_type=='acc':
        ax[0].set_ylabel('m/s')
        ax[0].set_title('acceleration')
    else:
        ax[0].set_ylabel('rad/s')
        ax[0].set_title('gyroscope')
        
    ax[0].set_xlabel('samples')
        

    

        
