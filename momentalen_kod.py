#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:45:38 2020

@author: dichoski
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:44:12 2020

@author: dichoski
"""
#%% Imports

from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
import pyeeg as pe
import pywt
import time
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter, welch
from scipy.integrate import simps
# from numpy.fft import ifft
# import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold #train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import f1_score

#%% Functions

############# Time Domain features #############

# 1. μ, mean of the raw signal over time N: np.mean()
# 2. σ, standard deviation of the raw signal: np.std()

# 3. δ, mean of the absolute values of the first differences of the raw signal:
def mean_abs_first_dif_raw(data):
    N = len(data)
    mean_abs = 0
    j=0
    for i in range(0,N-1):
        j=i+1
        #print(data[j],'-',data[i],'=')
        mean_abs = mean_abs +  abs(data[j]-data[i])
        #print(mean_abs)
    mean_abs = mean_abs/(N-1)
    return mean_abs

# 4. δ', mean of the absolute values of the first differences of the normalized signal:
def mean_abs_first_dif_norm(data):
    st_dev = np.std(data)
    if (st_dev == 0):
        return 0
    return mean_abs_first_dif_raw(data)/st_dev

# 5. γ, mean of the absolute values of the second differences of the raw signal:
def mean_abs_second_dif_raw(data):
    N = len(data)
    mean_abs = 0
    j=0
    for i in range(0,N-2):
        j=i+2
        #print(data[j],'-',data[i],'=')
        mean_abs = mean_abs +  abs(data[j]-data[i])
        #print(mean_abs)
    mean_abs = mean_abs/(N-2)
    return mean_abs

# 6. mean of the absolute values of the second differences of the normalized signal:
def mean_abs_second_dif_norm(data):
    st_dev = np.std(data)
    if (st_dev == 0):
        return 0
    return mean_abs_second_dif_raw(data)/st_dev


def rms(data):
    return np.sqrt(np.mean(np.square(data)))

def hoc(data):
    counter = 0
    mean_data = data - np.mean(data)
    for i in range(len(mean_data)-1):
        if mean_data[i]*mean_data[i+1] < 0:
            counter = counter + 1
    return counter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_time_features(data):
    feat = []

    feat.append(np.mean(data))
    # feat.append(rms(data))
    feat.append(np.std(data))
    feat.append(mean_abs_first_dif_raw(data))
    feat.append(mean_abs_first_dif_norm(data))
    feat.append(mean_abs_second_dif_raw(data))
    feat.append(mean_abs_second_dif_norm(data))
    feat.append(kurtosis(data))
    feat.append(skew(data))         # skewness
    # feat.append(np.var(data)) # variance/activity
    feat.append((pe.hjorth(data))[0]) # mobility
    feat.append((pe.hjorth(data))[1]) # complexity
    # feat.append(pe.hurst(data)) # Hurst Exponent
    
    # OVA NE, valjda e gresna funkcijata
    # alpha_beta = butter_bandpass_filter(data, 8, 32, 128, order=10)
    # feat.append(hoc(alpha_beta))
    
    # feat = np.asarray(feat)
    return feat # 10 features
    
############# Time Domain Features #############


############# FFT features #############
    
def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """

    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp
    
def get_frequency_features(data):
    # delta_power = bandpower(data,sample_rate,[0.5,4], relative=True)
    theta_power = bandpower(data,sample_rate,[4,  8], relative=False)
    alpha_power = bandpower(data,sample_rate,[8, 12], relative=False)
    beta_power  = bandpower(data,sample_rate,[12,30], relative=False)
    # gamma_power = bandpower(data,sample_rate,[30,45], relative=False)
    theta_power_rel = bandpower(data,sample_rate,[4,  8], relative=True)
    alpha_power_rel = bandpower(data,sample_rate,[8, 12], relative=False)
    beta_power_rel  = bandpower(data,sample_rate,[12,30], relative=True)
    # gamma_power_rel = bandpower(data,sample_rate,[30,45], relative=True)
    feat = []
    # feat.append(delta_power)
    feat.append(theta_power)
    feat.append(alpha_power)
    feat.append(beta_power)
    # feat.append(gamma_power)
    feat.append(theta_power_rel)
    feat.append(alpha_power_rel)
    feat.append(beta_power_rel)
    # feat.append(gamma_power_rel)

    return feat # 4 features

############# FFT features #############




# ############# Wavelet features #############
    
def get_wavelet_features(data):
    coeffs=pywt.wavedec(data,'db4',level=4)
    # cA4, cD4, cD3, cD2, cD1 = coeffs
    coeffs = coeffs[4:] # cA4 e delta
    feat = []
    
    # energy_total = np.sum([np.sum(np.square(x)) for x in coeffs])
    
    for x in coeffs:
        energy = np.sum(np.square(x))
        entropy = 0 - np.sum(np.square(x) * np.log(np.square(x)))
        # rms_val = rms(x)
        
        feat.append(energy)
        # feat.append(rms_val)
        feat.append(entropy)
        break
        
    return feat # 2 features * 4 bands = 8

# ############# Wavelet features #############
    

############# Multi‐Electrode Features #############
    
def get_multi_electrode_features(data):
    # data ima shape (32,512)
    
    # 0 so 29
    # 1 so 28
    # 2 so 27
    # 3 so 26
    # 4 so 25
    # 5 so 24
    # 6 so 23
    # 7 so 22
    # 8 so 21
    # 9 so 20
    # 10 so 19
    # 11 so 18
    # 13 so 17
    # 14 so 16
    feat = []
    band = [4,8,12,30,45] # 4 bands
    sym = [0,29,1,28,2,27,3,26,4,25,5,24,6,23,7,22,8,21,9,20,10,19,11,18,13,17,14,16]
    for i in range(len(sym)-1):
        if sym[i] not in channel:
            continue
        if i%2 != 0:
            continue
        powerL,_ = pe.bin_power(data[sym[i]][:], band, sample_rate)
        powerR,_ = pe.bin_power(data[sym[i+1]][:], band, sample_rate)
        dif_asym = powerR - powerL
        rat_asym = powerR / powerL
        for j in range (len(band)-1):
            feat.append(dif_asym[j])
            feat.append(rat_asym[j])
    
    return feat # 14 kanali * 4 bendovi * 2 features = 112

############# Multi‐Electrode Features #############    

    
def get_all_features (subject,window_size,step_size,sample_rate): # tuka doagja x['data'] # 40x40x8064
    # meta_video=
    for tr in range(0,40):
        print('Trial:',tr)
        trial = subject[tr,:32,384:] # bez prvite 3 sekundi i samo prvite 32 kanali
        # sega imame trial so shape (32,7680)
        
        # SKALIRANJE OD 0 DO 1
        # for ch in range (32): #(32,7680)
        #     trial[ch,:] = (trial[ch,:] - trial[ch,:].min()) / (trial[ch,:] .max() - trial[ch,:] .min())
        # trial = MinMaxScaler().fit_transform(trial.T).T
        
        meta_data = np.array([]) #meta vector for analysis
        for ch in range(32):
            # if ch not in channel:
            #     continue
            start = 0
            while start + window_size <= trial.shape[1]:
                # meta_array = []
                # print('Channel:',ch)
                X = trial[ch, start : start + window_size] # 512 dolzina                                    
                meta_data = np.append(meta_data, get_wavelet_features(X)
                                        # np.array(get_time_features(X) + get_frequency_features(X) + get_wavelet_features(X))
                                        # np.array(get_time_features(X) + get_frequency_features(X))
                                      )
                start = start + step_size
                
        # Multichannel
        # start = 0
        # while start + window_size <= trial.shape[1]:
        #     Z = trial[:,start : start + window_size]
        #     meta_data = np.append(meta_data, get_multi_electrode_features(Z))
        #     start = start + step_size
            
        if tr == 0:
            meta_video = meta_data.copy()[np.newaxis, :]
        else:
            meta_video = np.concatenate((
                meta_video, meta_data[np.newaxis, :]
                ), axis=0)
    return meta_video
    # 21 features * 32 channels * 29 windows


#%% Globals

path="/Users/dichoski/Desktop/data/"
path_save = '/Users/dichoski/Desktop/features/'
filenames = [f for f in listdir(path) if (isfile(join(path, f)) and '.dat' in f)]
filenames.sort()

band = [4,8,12,16,25,45] #5 bands
window_size = 512
step_size = 256
sample_rate = 128.0 #Sampling rate of 128 Hz
nSubj = 32
nTrial = 40
total = nSubj * nTrial
sym_channels = ['F3','F4','F7','F8','FC1','FC2','FC5','FC6','Fp1','Fp2']
channel = [0,2,3,4,5,24,25,26,27,29]
# channel = [1,2,3,4,6,11,13,17,19,20,21,25,29,31]
all_channels = ['Fp1', # 0
                'AF3', # 1
                'F7',  # 2
                'F3',  # 3
                'FC1', # 4
                'FC5', # 5
                'T7',  # 6 
                'C3',  # 7
                'CP1', # 8
                'CP5', # 9
                'P7',  # 10
                'P3',  # 11
                'Pz',  # 12
                'PO3', # 13
                'O1',  # 14
                'Oz',  # 15
                'O2',  # 16
                'PO4', # 17
                'P4',  # 18
                'P8',  # 19
                'CP6', # 20
                'CP2', # 21
                'C4',  # 22
                'T8',  # 23
                'FC6', # 24
                'FC2', # 25
                'F4',  # 26
                'F8',  # 27
                'AF4', # 28
                'Fp2', # 29
                'Fz',  # 30
                'Cz']  # 31
# Simetricni kanali:
# 0 so 29
# 1 so 28
# 2 so 27
# 3 so 26
# 4 so 25
# 5 so 24
# 6 so 23
# 7 so 22
# 8 so 21
# 9 so 20
# 10 so 19
# 11 so 18
# 13 so 17
# 14 so 16

# bez 12, 15,30,31


#%% Extracting features

print("Filenames are ", filenames)

print("\nLoading EEG files...")

i = 1
total_time = 0
for filename in filenames:
    start_time = time.process_time()
    feat = []
    with open(path + filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
    print('\nWorking on:', filename)
    subj_data = subject['data']
    feat = get_all_features(subj_data,window_size,step_size,sample_rate)
    np.save(path_save+'subject' + str(i), feat, allow_pickle=True, fix_imports=True)
    i = i + 1
    print(filename, 'done')
    total_time = total_time + time.process_time() - start_time
    print('Time taken:',time.process_time() - start_time) # 46 sek za 1 user
print('Processing done.')
print('Time taken:',total_time/60,'mins')



#%% Labels
# y treba da ima shape (32 subjects x 40 videos)
labels_val_total = []
labels_aro_total = []
for filename in filenames:
    with open(path + filename, 'rb') as f: 
        subject = pickle.load(f, encoding='latin1') #data,labels
    labels = subject['labels']
    labels_val = []
    labels_aro = []
    for tr in range(40):
        if labels[tr][0] >= 4.5:#5.25431250000001:
            kvadrant_val = 0
        else:
            kvadrant_val = 1
        if labels[tr][1] >= 4.5:#5.156710937500021:
            kvadrant_aro = 0
        else:
            kvadrant_aro = 1
        labels_val.append(kvadrant_val)
        labels_aro.append(kvadrant_aro)
    labels_val_total.append(labels_val)
    labels_aro_total.append(labels_aro)
    
total = nSubj * nTrial    
labels_val_total = np.reshape(np.asarray(labels_val_total),total)
labels_aro_total = np.reshape(np.asarray(labels_aro_total),total)



#%%
subject_list = []
for i in range(1,33): 
    subject = np.load(path_save+'subject'+str(i)+'.npy',allow_pickle=True)
    subject_list.append(subject)
    
subject_list=np.asarray(subject_list)
subject_list=np.reshape(subject_list,((40*32,subject_list.shape[2]))) 


#%% Training and Testing

# samo prvite 22
# subject_list_new = subject_list[:880,:]
# labels_aro_total_new = labels_aro_total[:880]
# X_train = subject_list_new[:720,:]
# X_test = subject_list_new[-160:,:]
# y_train = labels_aro_total_new[:720]
# y_test = labels_aro_total_new[-160:]
#########

n_test_subj = 4
test_len = n_test_subj * 40 #160
train_len = (32 - n_test_subj) * 40 #1120


train_valid_data   = subject_list[:train_len] # (1120, 34684)
train_valid_labels = labels_val_total[:train_len]

# Test data ja oddeluvame sega i ne ja 'gibame' do kraj!!!
test_data   = subject_list[-test_len:] # (160, 34684)
test_labels = labels_val_total[-test_len:] 

X = train_valid_data
y = train_valid_labels

A = test_data
b = test_labels
#%%


groups = np.empty(train_valid_data.shape[0]) # 28 grupi
k = 0
l = 1
for i in range(28):
  for j in range(40):
    groups[k] = l
    k = k + 1
  l = l + 1


  
gkf = GroupKFold(n_splits=7) # 7 * 4 = 28
# for train_index, val_index in gkf.split(X, y, groups):
#     X_train, X_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]


#%%
n_niza = []
for i in range(3,40):
    if i%2==0:
        continue
    n_niza.append(i)
    
param_grid = { #'reduce_dim__n_components': [60, 80, 100],
                'n_neighbors': n_niza,
                #'knn__weights': ['uniform','distance'],
                #'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
             }

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('knn', KNeighborsClassifier(algorithm='brute'))
        ])

gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', cv=gkf, return_train_score=True, verbose=10)
gs.fit(X, y, groups)    
     

    


param_grid_svc = { 'C': [0.001, 0.01, 0.1, 1, 10],
                   'gamma' : [0.001, 0.01, 0.1, 1],
                   'kernel' : ['rbf']
                  }
gs_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, scoring='accuracy', cv=gkf, return_train_score=True, verbose=10)
gs_svc.fit(X, y, groups)    
    
    
    
#%%

# This function is used to check the accuracy score for a given model, training and testing data
def get_score(model,X_train,X_test,y_train,y_test): 
	model.fit(X_train,y_train)
	return model.score(X_test,y_test)   

train_x = meta.reshape(1280,504*6)
train_y = labels_val_total
  
kf=KFold(n_splits=10)
for train_index,test_index in kf.split(train_x):
	X_train,X_test,y_train,y_test=train_x[train_index],train_x[test_index],train_y[train_index],train_y[test_index]    
   
clf = KNeighborsClassifier(n_neighbors=13) #knn model for classifying the valence
predicted_val=get_score(clf,X_train,X_test,y_train,y_test)
print(predicted_val)
# so 9 sosedi: 0.6796875
# so 11 sosedi: 0.6875
# so 13 sosedi: 0.6796875
# so 17-27 sosedi: 0.640625
# so 27-139 sosedi: 0.6484375

train_a = labels_aro_total

kf1=KFold(n_splits=8)
for train_index,test_index in kf1.split(train_x):
	X_train1,X_test1,y_train1,y_test1=train_x[train_index],train_x[test_index],train_a[train_index],train_a[test_index]

clf1 = KNeighborsClassifier(n_neighbors=9) #knn model for classifying the arousal
arousal_val=get_score(clf1,X_train1,X_test1,y_train1,y_test1)
print(arousal_val)
# so bilo koj broj sosedi: 0.6171875

######### Support Vecror Machine #########
clf_svc = SVC(C=0.1, kernel = 'poly', random_state = 42, gamma='auto')
clf_svc.fit(X_train, y_train)
y_predict = clf_svc.predict(X_test)
print("Accuracy score of Valence:")
print(accuracy_score(y_test, y_predict)*100)
# Accuracy score of Valence:
# 63.74999999999999


clf1_svc = SVC(C=0.1, kernel = 'rbf', random_state = 42, gamma='scale')
clf1_svc.fit(X_train1, y_train1)
y_predict1 = clf_svc.predict(X_test1)
print("Accuracy score of Arousal:")
print(accuracy_score(y_test1, y_predict1)*100)
# Accuracy score of Arousal:
# 64.0625


######### Gradient Boosting Classifier #########
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X_train, y_train)
accuracy_gb = clf_gb.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_gb)
# Accuracy score of Valence: 0.58125


######### AdaBoost Classifier #########
clf_ab = AdaBoostClassifier()
clf_ab.fit(X_train, y_train)
accuracy_ab = clf_ab.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_ab)
# Accuracy score of Valence: 0.56875

 
######### Decision Tree Classifier ######### 
clf_dt = DecisionTreeClassifier(random_state=42)   
clf_dt.fit(X_train, y_train)
accuracy_dt = clf_dt.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_dt) 
# Accuracy score of Valence: 0.59375

 
######### Random Forest Classifier #########
clf_rf = RandomForestClassifier(n_estimators=350)   
clf_rf.fit(X_train, y_train)
accuracy_rf = clf_rf.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_rf) 
# Accuracy score of Valence: 0.60625 so 100 drva
# Accuracy score of Valence: 0.625   so 150 drva
# Accuracy score of Valence: 0.6625  so 200 drva
# Accuracy score of Valence: 0.66875 so 350 drva <<<<<<<<<<
# Accuracy score of Valence: 0.64375 so 400 drva
# Accuracy score of Valence: 0.6125  so 450 drva

######### Linear Discriminant Analysis #########
clf_lda = LinearDiscriminantAnalysis()   
clf_lda.fit(X_train, y_train)
accuracy_lda = clf_lda.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_lda) 
# Accuracy score of Valence: 0.4625
    

######### Extreme Gradient Boosting Classifier #########
clf_xgb = xgb.XGBClassifier(n_estimators=100)       
clf_xgb.fit(X_train, y_train)

accuracy_xgb = clf_xgb.score(X_test,y_test)
print('Accuracy score of Valence:', accuracy_xgb) 
# Accuracy score of Valence: 0.625
                            
                            
preds = clf.predict(X_test)
print("Precision = {}".format(precision_score(y_test, preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, preds)))  
# Precision = 0.5616883116883117
# Recall = 0.5385395537525355
# Accuracy = 0.625 
                            
# f1 score



#%% CNN

from EEGModels import EEGNet

model = EEGNet(nb_classes = 2, Chans = 32, Samples = 128)

#CNN NE









                            
                            