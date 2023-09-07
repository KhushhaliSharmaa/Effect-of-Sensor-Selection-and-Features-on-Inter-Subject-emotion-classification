# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:35:41 2022

@author: hp
"""

#Library
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.integrate import simps
from scipy.signal import filtfilt, butter, lfilter, welch
import scipy.stats as sstats
import nolds
import pickle
import matplotlib.pyplot as plt
import antropy as ant
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%%
nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 32, 8064  
no_of_users=32
fout_data = open('data_preprocessed_python.dat','w')

# data_dict = {}
labels_dict = {}
# features_dict = {}
# filter_dict = {}
# CAR_dict = {}
# PSD_dict = {}
# Bandpower_dict = {}
# Psdband_dict = {}
# Mean_dict = {}
# Stdev_dict = {}
# Skew_dict = {}
# Median_dict = {}
# Kurt_dict = {}
# Ent_dict = {}
# HFD_dict = {}
# Corr_dict = {}
# Var_dict = {}
# data_arr = []
# labels_arr = []
# CAR_arr = []
Feature_array = []
# CAR_all = []
# Feature_trial = []
# BP_Power = []
# FAA_trial = []
PCA_chunk, subject_all = [], []
Label_1, Label_2, Label_3, Label_4 = [] , [] , [] , []
#%%
#LOOP FOR 32 SUBJECTS
for i in range(no_of_users):#nUser  #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:			
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
                
            fname = "s"+str(name)+".dat"
            x = pickle.load(open(fname, 'rb'), encoding='latin1')
            data = x['data']
            labels = x['labels']
            
            #LOOP FOR 40 TRIALS
            for k in range(nTrial):
                trial = data[k] #For single trial
                EEG_Trial = trial[0:32,:] #Selecting 32 EEG-sensor
                #EEG_Sensor = np.delete(EEG_Trial, [1,2,3,5,8,9,11,12,18,19,21,23,24,26,27,29,31], axis=0) #15 Channel Selection
                #EEG_Sensor = EEG_Trial[[0,7,15,16,30]] #5-SENSOR SELECTION
                EEG_Sensor = EEG_Trial[[0,7,13,14,15,16,20,25,28,30]] #10-SENSOR
                Ctrial = EEG_Sensor - np.mean(EEG_Sensor, axis=0) #Common Average Referencing
                C_EEG_Trial = Ctrial[0:10, 384:8064] #3sec baseline removal
                #C_EEG_Trial = Ctrial[0:32, 4224:8064] #3sec baseline removal
                C_Baseline_EEG = Ctrial[0:10, 0:384] #Extracting baseline
                bp_theta_Value, bp_alpha_Value, bp_beta_Value , bp_gamma_Value  = [] , [] , [] , []
                Mean_Value, Stdev_Value, Skew_Value, Median_Value, Kurt_Value, Ent_Value, HFD_Value, Corr_Value = [] , [] , [] , [] , [], [], [], []

                
                #LOOP FOR 15 SENSORS
                for K in range(np.shape(C_EEG_Trial[:,1])[0]):
                    # Filtering
                      order = 5
                      sample_freq = 128
                      cutoff_freq = 2
                      sample_duration = 60
                      no_of_samples = sample_freq * sample_duration
                      time = np.linspace(0, sample_duration, no_of_samples, endpoint = False)
                      normalized_cutoff = 2 * cutoff_freq / sample_freq
                      b_filt, a_filt = scipy.signal.butter(order, normalized_cutoff, analog=False)
                      filtered_signal = scipy.signal.lfilter(b_filt, a_filt, C_EEG_Trial[0:10,:], axis = 0)
                      filt_window = np.hsplit(filtered_signal, 60)      
                      
                      #Compute PSD
                      sf = 128
                      time = np.arange(C_EEG_Trial.size)/ sf  
                      win = 1 * sf #Define window lenght(1 second)
                      freqs_filt, psd_filt = freqs, Psd = scipy.signal.welch(filtered_signal, fs=128.0, nperseg=win, axis=1)
                      freq_filt_res = freqs[1] - freqs[0]
                      
                      #BANDPASS FILTER
                      nyq = 0.5 * sample_freq
                      
                      #FOR THETA
                      fmin_theta = 3
                      fmax_theta = 7
                      low_theta = fmin_theta / nyq
                      high_theta = fmax_theta / nyq
                      b_theta, a_theta = scipy.signal.butter(order, [low_theta, high_theta], btype='band', analog=False)
                      filtered_theta = scipy.signal.lfilter(b_theta, a_theta, C_EEG_Trial[0:10,:], axis = 0)
                      filt_theta = np.hsplit(filtered_theta, 60)
                     
                      
                      #FOR ALPHA
                      fmin_alpha = 8
                      fmax_alpha = 13
                      low_alpha = fmin_alpha / nyq
                      high_alpha = fmax_alpha / nyq
                      b_alpha, a_alpha = scipy.signal.butter(order, [low_alpha, high_alpha], btype='band', analog=False)
                      filtered_alpha = scipy.signal.lfilter(b_alpha, a_alpha, C_EEG_Trial[0:10,:], axis = 0)
                      filt_alpha = np.hsplit(filtered_alpha, 60)
                      
                      
                      #FOR BETA
                      fmin_beta = 14
                      fmax_beta = 29
                      low_beta = fmin_beta / nyq
                      high_beta = fmax_beta / nyq
                      b_beta, a_beta = scipy.signal.butter(order, [low_beta, high_beta], btype='band', analog='False')
                      filtered_beta = scipy.signal.lfilter(b_beta, a_beta, C_EEG_Trial[0:10,:], axis = 0)
                      filt_beta = np.hsplit(filtered_beta, 60)
                      # #FOR Frontal assymetery of beta band
                      # freqs_b, Psd_b = scipy.signal.welch(filtered_beta, nperseg=win, axis=1)
                      # idx_b = np.logical_and(freqs_filt >= fmin_beta, freqs_filt <= fmax_beta)
                      # pow_rightbeta = simps(Psd_b[19,:][idx_b], dx=freq_filt_res)
                      # pow_leftbeta = simps(Psd_b[2,:][idx_b], dx=freq_filt_res)
                      # #Computing FAA at F3 & F4 channel
                      # FAA_beta = np.log(pow_rightbeta) - np.log(pow_leftbeta)

                      #FOR GAMMA
                      fmin_gamma = 30
                      fmax_gamma = 47
                      low_gamma = fmin_gamma / nyq
                      high_gamma = fmax_gamma / nyq
                      b_gamma, a_gamma = scipy.signal.butter(order, [low_gamma, high_gamma], btype='band', analog='False')
                      filtered_gamma = scipy.signal.lfilter(b_gamma, a_gamma, C_EEG_Trial[0:10,:], axis = 0)
                      filt_gamma = np.hsplit(filtered_gamma, 60)
#%%
                      #Loop for 60 chunks
                      #Compute PSD, bandpower
                      for j in range (60):
                          freqs, Psd = scipy.signal.welch(((filt_window[j])[K]),fs=128, nperseg = win, axis=0 )
                          freqs_theta, Psd_theta = scipy.signal.welch(((filt_theta[j])[K]), fs=128, nperseg = win, axis=0)
                          freqs_alpha, Psd_alpha = scipy.signal.welch(((filt_alpha[j])[K]), fs=128, nperseg = win, axis=0)
                          freqs_beta, Psd_beta = scipy.signal.welch(((filt_beta[j])[K]), fs=128, nperseg = win, axis=0)
                          freqs_gamma, Psd_gamma = scipy.signal.welch(((filt_gamma[j])[K]), fs=128, nperseg = win, axis=0)
                          freqs_res = freqs[1] - freqs[0]
                          idx_theta = np.logical_and(freqs >= fmin_theta, freqs <= fmax_theta)
                          idx_alpha = np.logical_and(freqs >= fmin_alpha, freqs <= fmax_alpha)
                          idx_beta = np.logical_and(freqs >= fmin_beta, freqs <= fmax_beta)
                          idx_gamma = np.logical_and(freqs >= fmin_gamma, freqs <= fmax_gamma)
                          bp_theta = simps(Psd_theta[idx_theta], dx=freqs_res)
                          bp_alpha = simps(Psd_alpha[idx_alpha], dx=freqs_res)
                          bp_beta = simps(Psd_beta[idx_beta], dx=freqs_res)
                          bp_gamma = simps(Psd_gamma[idx_gamma], dx=freqs_res)
                          bp_theta_Value.append(bp_theta)
                          bp_alpha_Value.append(bp_alpha)
                          bp_beta_Value.append(bp_beta)
                          bp_gamma_Value.append(bp_gamma)
#%%             #FEATURE EXTRACTION            
                n = len(filt_window)
                for l in range (n):
                    for m in range (len(filt_window[l])):
                        # w_mean = np.mean((filt_window[l])[m])
                        # w_stdev = np.std((filt_window[l])[m])
                        w_skew = sstats.skew((filt_window[l])[m])
                        w_median = np.median((filt_window[l])[m])
                        w_kurt = sstats.kurtosis((filt_window[l])[m])
                        # ent = ant.sample_entropy((filt_window[l])[m])
                        # hfd = ant.higuchi_fd((filt_window[l])[m])
                        # corr_dim = nolds.corr_dim(((filt_window[l])[m]), emb_dim=2)
                        # Mean_Value.append(w_mean)
                        # Stdev_Value.append(w_stdev)
                        Skew_Value.append(w_skew)
                        Median_Value.append(w_median)
                        Kurt_Value.append(w_kurt)
                        # Ent_Value.append(ent)
                        # HFD_Value.append(hfd)
                        # Corr_Value.append(corr_dim)
                # FAA_trial.append(FAA_beta)
                # frontal_assym = np.array(FAA_trial)
                Feature_array.append([Skew_Value, Kurt_Value, bp_gamma_Value, bp_theta_Value, bp_alpha_Value, bp_beta_Value])
                Feature = np.array(Feature_array)
                Feature_matrix = np.reshape(Feature, (Feature.shape[0], Feature.shape[1]*Feature.shape[2]))
                # Feature_matrix = np.column_stack((Feature_reshape, frontal_assym))
                df_Feature = pd.DataFrame(Feature_matrix)
                df_Feature[df_Feature==np.inf]=np.nan
                df_Feature.fillna(df_Feature.mean(), inplace=True)
                scaler=StandardScaler()
                Feature_rescaled = scaler.fit_transform(Feature_matrix)
Feature_chunk = np.hsplit(Feature_rescaled, 60)
# x = len(Feature_chunk)
# for p in range(x):
#     pca = PCA(n_components = 0.95)
#     pca.fit(Feature_chunk[p])
#     PCA_Feature = pca.transform(Feature_chunk[p])
#     PCA_chunk.append(PCA_Feature)
# for w in range(len(PCA_chunk)):
#     for z in range(len(PCA_chunk[w])):
#         subject = np.vsplit(PCA_chunk[w], 32)
#     subject_all.append(subject)
#%%
