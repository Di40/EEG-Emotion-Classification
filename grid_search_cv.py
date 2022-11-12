#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:26:43 2020

@author: dichoski
"""
#%%
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
# from numpy.fft import ifft
# import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
# from sklearn.model_selection import KFold #train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.decomposition import PCA
# import pandas as pd
# import pymrmr

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

#%%
path="/Users/dichoski/Desktop/data/"
path_save = '/Users/dichoski/Desktop/out_final/site_features_scaled_0do1/'
filenames = [f for f in listdir(path) if (isfile(join(path, f)) and '.dat' in f)]
filenames.sort()
print("Filenames are ", filenames)
window_size = 512 # 1sec
start = 0
step_size = 256 # non-overlapping window
# sym_channels = ['F3','F4','F7','F8','FC1','FC2','FC5','FC6','Fp1','Fp2']
# channel = [1,2,3,4,6,11,13,17,19,20,21,25,29,31]
band = [4,8,12,16,25,45] #5 bands
window_size = 512 #Averaging band power of 2 sec
step_size = 256
sample_rate = 128.0 #Sampling rate of 128 Hz
nSubj = 32
nTrial = 40
total = nSubj * nTrial

#%% Labels
# y treba da ima shape (32 subjects x 40 videos)
labels_val_total = []
labels_aro_total = []
for filename in filenames:
    with open(path + filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
    labels = subject['labels']
    labels_val = []
    labels_aro = []
    for tr in range(40):
        if labels[tr][0] >= 4.5:
            kvadrant_val = 1
        else:
            kvadrant_val = 0
        if labels[tr][1] >= 4.5:
            kvadrant_aro = 1
        else:
            kvadrant_aro = 0
        labels_val.append(kvadrant_val)
        labels_aro.append(kvadrant_aro)
    labels_val_total.append(labels_val)
    labels_aro_total.append(labels_aro)
       
labels_val_total = np.reshape(np.asarray(labels_val_total),total)
labels_aro_total = np.reshape(np.asarray(labels_aro_total),total)


subject_list = []
for i in range(1,33): 
    subject = np.load(path_save+'subject'+str(i)+'.npy')
    subject_list.append(subject)
subject_list=np.asarray(subject_list)
subject_list=np.reshape(subject_list,((total,subject_list.shape[2]))) 

#%% Labels 4 class
# y treba da ima shape (32 subjects x 40 videos)
labels_4_total = []
for filename in filenames:
    with open(path + filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
    labels = subject['labels']
    labels_4 = []
    for tr in range(40):
        if labels[tr][0] >= 4.5 and labels[tr][1] >= 4.5:
            kvadrant = 1
        elif labels[tr][0] < 4.5 and labels[tr][1] >= 4.5:
            kvadrant = 2
        elif labels[tr][0] < 4.5 and labels[tr][1] < 4.5:
            kvadrant = 3
        elif labels[tr][0] >= 4.5 and labels[tr][1] < 4.5:
            kvadrant = 4
        labels_4.append(kvadrant)
    labels_4_total.append(labels_4)
    
total = nSubj * nTrial    
labels_4_total = np.reshape(np.asarray(labels_4_total),total)


# subject_list = []
# for i in range(1,33): 
#     subject = np.load(path_save+'subject'+str(i)+'.npy')
#     subject_list.append(subject)
# subject_list=np.asarray(subject_list)
# subject_list=np.reshape(subject_list,((total,subject_list.shape[2]))) 


# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

labels_4_total = label_encoder.fit_transform(labels_4_total)


#%% Labels 9 class
# y treba da ima shape (32 subjects x 40 videos)
labels_9_total = []
for filename in filenames:
    with open(path + filename, 'rb') as f: subject = pickle.load(f, encoding='latin1') #data,labels
    labels = subject['labels']
    labels_9 = []
    for tr in range(40):
        if labels[tr][0] >= 7 and labels[tr][1] >= 7:
            kvadrant = 'HVHA'
        elif labels[tr][0] >= 7 and labels[tr][1] > 4 and labels[tr][1] < 7:
            kvadrant = 'HVNA'
        elif labels[tr][0] >= 7  and labels[tr][1] <= 4:
            kvadrant = 'HVLA'
        elif labels[tr][0] > 4 and labels[tr][0] < 7 and labels[tr][1] >= 7:
            kvadrant = 'NVHA'
        elif labels[tr][0] > 4 and labels[tr][0] < 7  and labels[tr][1] > 4 and labels[tr][1] < 7:
            kvadrant = 'NVNA'
        elif labels[tr][0] > 4 and labels[tr][0] < 7  and labels[tr][1] <= 4:
            kvadrant = 'NVLA'
        elif labels[tr][0] <= 4 and labels[tr][1] >= 7:
            kvadrant = 'LVHA'
        elif labels[tr][0] <= 4 and labels[tr][1] > 4 and labels[tr][1] < 7:
            kvadrant = 'LVNA'
        elif labels[tr][0] <= 4 and labels[tr][1] <= 4:
            kvadrant = 'LVLA'
        labels_9.append(kvadrant)
    labels_9_total.append(labels_9)
    
total = nSubj * nTrial    
labels_9_total = np.reshape(np.asarray(labels_9_total),total)


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

labels_9_total = label_encoder.fit_transform(labels_9_total.reshape(-1,1))




#%%

n_test_subj = 4 
test_len = n_test_subj * 40
train_len = (32 - n_test_subj) * 40


train_valid_data   = subject_list_minmax[:train_len] # (1120, 34684)
train_valid_labels = labels_aro_total[:train_len]

# Test data ja oddeluvame sega i ne ja 'gibame' do kraj!!!
test_data   = subject_list_minmax[-test_len:] # (160, 34684)
test_labels = labels_aro_total[-test_len:] 

groups = np.empty(train_valid_data.shape[0]) # 28 grupi
k = 0
l = 1
for i in range(28):
  for j in range(40):
    groups[k] = l
    k = k + 1
  l = l + 1

X = train_valid_data
y = train_valid_labels

n_splits=7
gkf = GroupKFold(n_splits=n_splits) 


#%%

param_grid_svc = { 'C': [1000,2000,3000,4000,5000,10000,50000,100000,1000000],
                   'kernel' : ['rbf']
                  }
gs_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, cv=gkf, scoring='accuracy', verbose=10, n_jobs=-1)
gs_svc.fit(X,y,groups)    
print(gs_svc.score(test_data, test_labels))
print(gs_svc.best_params_)

print(confusion_matrix(test_labels,gs_svc.predict(test_data)))
print(confusion_matrix(test_labels,gs_svc.predict(test_data),normalize='true'))


'''
Best estimator: SVC(C=0.001, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
Best Score: 0.6303571428571428 (accuracy)
'''


'''
- site_pomalku_scaled: 
                        {'C': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}
                        Best Score: 0.6303571428571428 (f1_micro)
                        Test data score: 0.6375
                        
- site_features_scaled_0do1:
                            {'C': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}
                            Best Score: 0.6303571428571428 (f1_micro)
                            Test data score: 0.6375
    
'''


#%%
param_grid_rf = { 
              'n_estimators': [500, 1000],
              'max_features': ['log2'],
              'max_depth' : [6,7,8],
               'criterion' :['entropy']
              }
gs_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, scoring='accuracy', cv=gkf, verbose=10,n_jobs=-1)
gs_rf.fit(X, y, groups)    
gs_rf.score(test_data, test_labels)

'''
Best Estimator: RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='entropy', max_depth=7, max_features='log2',
                                       max_leaf_nodes=None, max_samples=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=500,
                                       n_jobs=None, oob_score=False, random_state=None,
                                       verbose=0, warm_start=False)
Best Score: 0.6366071428571428 (f1_micro)    
'''

'''
- site_pomalku_scaled: 
                      {'criterion': 'entropy',
                       'max_depth': 7,
                       'max_features': 'sqrt',
                       'n_estimators': 500}
                     Best score: 0.6321428571428571 (f1_micro)
                     Score na test data: 0.6375
- site_features_scaled_0do1:
                            {'C': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}
                            Best Score: 0.6321428571428571 (f1_micro)
                            Test data score: 0.6375
'''
# gs_rf.get_support()
# selected_feat= X_train.columns[(gs_rf.get_support())]
# print(len(selected_feat))

# pd.series(gs_rf.estimator_,feature_importances_,.ravel()).hist()

selector = SelectFromModel(RandomForestClassifier(criterion='entropy',max_depth=7,max_features='sqrt',n_estimators=500))
gkf = GroupKFold(n_splits=7) # 7 * 4 = 28
for train_index, val_index in gkf.split(X, y, groups):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
selector.fit(X_train, y_train)
selector.get_support()



#%%
param_grid_ab = { 
                 'n_estimators': [50, 100],
                'learning_rate' : [0.01,0.05,0.1,1],
                }
gs_ab = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ab, scoring='accuracy', cv=gkf, verbose=10, n_jobs=-1)
gs_ab.fit(X, y, groups)
gs_ab.score(test_data, test_labels)

'''
Best Estimaror: AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.01,
                                   n_estimators=100, random_state=None)
Best Score: 0.6321428571428571 (f1_micro)


- site_pomalku_scaled: 
                      {'learning_rate': 0.1, 'n_estimators': 100}
                      Best score: 0.6205357142857143
                      Score na test data: 0.6625
'''

#%%

param_grid_knn = {
                    "n_neighbors": [3,7,13,29,57,87,135]#181,183,185,187,189]
                }
gs_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, scoring='accuracy', cv=gkf, verbose=10,n_jobs=-1)
gs_knn.fit(X, y, groups)
gs_knn.score(test_data, test_labels)
#0.40625
'''
Best Estimator: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                     metric_params=None, n_jobs=None, n_neighbors=57, p=2,
                                     weights='uniform')
Best Score: 0.6321428571428571 (f1_micro)



- site_pomalku_scaled: 
                      {'n_neighbors': 57}
                      Best score: 0.6330357142857143
                      Score na test data: 0.6375

'''

#%%
classifiers = {
        "Gradient Boosting" : GradientBoostingClassifier(),
        "AdaBoost" : AdaBoostClassifier(),
        "Random Forest" : RandomForestClassifier(),
        "Extreme Gradient Boosting" : xgb.XGBClassifier(),
        "Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        }

params = {
        "Gradient Boosting" : [{ "loss" : ["deviance"],
                               "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                               "min_samples_split": np.linspace(0.1, 0.5, 12),
                               "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                               "max_depth":[3,5,8],
                               "max_features":["log2","sqrt"],
                               "criterion": ["friedman_mse",  "mae"],
                               "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                               "n_estimators":[10]
                               }],
        "Nearest Neighbors": [{"n_neighbors": [3,7,13,29,57,87,135,181,183,185,187,189]}],
        "SVM": [{ "kernel": ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                  "C": [0.001, 0.01, 0.1, 1, 10]
               }],

        "Decision Tree": [{  { 'criterion':['gini','entropy'],
                               'max_depth': np.arange(3, 15)}
                         }],
        "Random Forest": [{"n_estimators": [10, 50, 100, 250, 500, 1000]}],
        "Logistic Regression": [{"C": np.logspace(-2, 3, 6).tolist()}],
        "Naive Bayes": [] # nema hiper parametri
        }
gs = GridSearchCV(estimator=classifiers, param_grid=params, scoring='‘f1_micro', cv=gkf, return_train_score=False, verbose=10)
gs.fit(X, y, groups)                   
                
                

#%%
xgb = XGBClassifier(objective='binary:logistic',nthread=4,seed=42)
params = {
    'n_estimators': [500,1000,1500,2000],
    'learning_rate': [0.1, 0.01]
}


gs_xgb = GridSearchCV(estimator=xgb, param_grid=params, scoring='neg_log_loss', verbose=10, n_jobs = -1)
gs_xgb.fit(X,y)
gs_xgb.score(A,b)


'''
- site_pomalku_scaled: 
                        {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 60}
                        Best Score: 0.625 (f1_micro)
                        Test data score: 0.6375
                        
                        
                        
'''                        
#%%   
gkf = GroupKFold(n_splits=7) # 7 * 4 = 28
for train_index, val_index in gkf.split(X, y, groups):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
ab=AdaBoostClassifier(learning_rate= 0.1, n_estimators= 100)
ab.fit(X_train,y_train)
# import some data to play with
class_names = ["LOW","HIGH"]

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
xgb= XGBClassifier(learning_rate= 0.01, max_depth= 3, n_estimators= 60)
xgb.fit(X,y)
for title, normalize in titles_options:
    disp = plot_confusion_matrix(ab, test_data, test_labels, # classifier
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

#%%
X_test, y_test= test_data, test_labels
y_score = ab.decision_function(X_test) #classifier
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


# from sklearn.metrics import precision_recall_curve


disp = plot_precision_recall_curve(ab, X_test, y_test) # classifier
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))



#%%
from sklearn.metrics import precision_recall_fscore_support

y_pred = ab.predict(X_test) #classifier
y_true = y_test

print('Macro:')
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print('Precision:',precision,'Recall:',recall,'F1:',f1)

print('Micro:')
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
print('Precision:',precision,'Recall:',recall,'F1:',f1)

print('Weighted:')
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print('Precision:',precision,'Recall:',recall,'F1:',f1)

'''
XGB:
Macro:
Precision: 0.31875 Recall: 0.5 F1: 0.3893129770992366
Micro:
Precision: 0.6375 Recall: 0.6375 F1: 0.6375
Weighted:
Precision: 0.40640624999999997 Recall: 0.6375 F1: 0.4963740458015267



AB:
Macro:
Precision: 0.6461538461538461 Recall: 0.5963488843813387 F1: 0.5924764890282131
Micro:
Precision: 0.675 Recall: 0.675 F1: 0.675
Weighted:
Precision: 0.6588461538461539 Recall: 0.675 F1: 0.6429075235109718
'''





#%%
from sklearn.model_selection import cross_val_score
svm = SVC()
parameters = {'kernel':('linear', 'rbf'),
              'C':(1,0.25,0.5,0.75),
              'gamma': (1,2,3,'auto'),
              'decision_function_shape':('ovo','ovr'),
              'shrinking':(True,False)}
clf = GridSearchCV(svm, parameters)
clf.fit(X,y)
print("accuracy:"+str(np.average(cross_val_score(clf, test_data, test_labels, scoring='accuracy'))))
print("f1:"+str(np.average(cross_val_score(clf, test_data, test_labels, scoring='f1'))))



#%%
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()

param_grid_svc = { 'C': [1,10,100],
                   'gamma' : [0.001, 0.1],
                   'kernel' : ['rbf']
                  }
gs_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, scoring='accuracy', cv=logo, return_train_score=True, verbose=10, n_jobs=-1)
gs_svc.fit(X, y, groups)    
gs_svc.score(test_data, test_labels)






               