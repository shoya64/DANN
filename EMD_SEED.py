#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:58:38 2020

@author: furukawashouya
"""


import scipy.io
#data_被験者番号_実験回数
data_1_1 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/1_20131027.mat')
data_1_2 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/1_20131030.mat')
data_1_3 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/1_20131107.mat')


#de_被験者番号_実験回数[映画番号-1](62*200*5)
de_1_1 = []
for i in range(1, 16):
    de_1_1.append(data_1_1['de_LDS' + str(i)])
de_1_2 = []
for i in range(1, 16):
    de_1_2.append(data_1_2['de_LDS' + str(i)])
de_1_3 = []
for i in range(1, 16):
    de_1_3.append(data_1_3['de_LDS' + str(i)])
    

#感情ラベル
label_data = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/label.mat')

import torch.nn as nn
import torch.optim as optim
import numpy as np
#import torch.nn.functional as F

from DANN import DANNM
from DANN import DANN_Bi

from matplotlib import pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

from torchsummary import summary


#データ作成
def window_slice(data, time_steps):#dataをtimestepsの長さずつに一つずつデータをずらして小分けする
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    xs = []
    for i in range(data.shape[0] - time_steps):
        xs.append(data[i: i + time_steps])
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs

def window_slice_MinMax(data, time_steps):
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    
    mm = preprocessing.MinMaxScaler()
    data_min_max = mm.fit_transform(data)
    xs = []
    for i in range(data_min_max.shape[0] - time_steps + 1):
        xs.append(data_min_max[i: i + time_steps])
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs


def setData_imp(data, label_data, time_steps):
    label = label_data['label']
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    data_1 = []
    data_2 = []
    for i in range(15):
        if i < 9:
            data_1.append(data[i])
        else :
            data_2.append(data[i])
    Y_train, Y_test = train_test_split(label.reshape(15,), test_size = 6, shuffle = False)
    for data, label in zip(data_1, list(Y_train)):
        X_train.append(window_slice_MinMax(data, time_steps))
        y_train.extend([label] * len(X_train[-1]))
        
    for data, label in zip(data_2, list(Y_test)):
        X_test.append(window_slice_MinMax(data, time_steps))
        y_test.extend([label] * len(X_test[-1]))
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    plus1 = np.array([1]*y_train.shape[0])
    plus2 = np.array([1]*y_test.shape[0])
    y_train = y_train + plus1
    y_test = y_test + plus2
    
    
    #ドメインラベル,1:train,0:test
    domain_train = np.array([1]*y_train.shape[0])
    domain_test = np.array([0]*y_test.shape[0])
    
    return np.concatenate(X_train), np.concatenate(X_test), y_train, y_test, domain_train, domain_test


#バッチ生成
def mkRandomBatch(train_x, train_t, domain, batch_size):
    
    idx = np.random.randint(0, train_x.shape[0] - batch_size)
    batch_x = train_x[idx:batch_size+idx, :, :]
    batch_t = train_t[idx:batch_size+idx]
    batch_domain = domain[idx:batch_size + idx]
    
    return torch.tensor(batch_x), torch.tensor(batch_t), torch.tensor(batch_domain)

def mkBatch(train_x, train_t, domain, batch_size):
    b_x = []
    b_t = []
    d = []
    
    for i in range(int(train_x.shape[0]/batch_size)):
        b_x.append(train_x[i*batch_size: batch_size*(i+1), :, :])
        b_t.append(train_t[i*batch_size: batch_size*(i+1)])
        d.append(domain[i*batch_size: batch_size*(i+1)])

    return b_x, b_t, d

import random
def rand_ints_nodup(a, b, k):
  ns = []
  while len(ns) < k:
    n = random.randint(a, b)
    if not n in ns:
      ns.append(n)
  return ns   
   
from visdom import Visdom


from PyEMD import EMD
s = np.random.random(100)
emd = EMD()
IMFs = emd(s, max_imf = 1)



"""
X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
X_train_2, X_test_2, Y_train_2, Y_test_2, domain_train_2, domain_test_2 = setData_imp(de_1_2, label_data, 50)
X_train_3, X_test_3, Y_train_3, Y_test_3, domain_train_3, domain_test_3 = setData_imp(de_1_3, label_data, 50)

X_train_a = np.concatenate([X_train_1, X_train_2, X_train_3], axis = 0)
X_test_a = np.concatenate([X_test_1, X_test_2, X_test_3], axis = 0)
Y_train_a = np.concatenate([Y_train_1, Y_train_2, Y_train_3], axis = 0)
Y_test_a = np.concatenate([Y_test_1, Y_test_2, Y_test_3], axis = 0)
domain_train_a = np.concatenate([domain_train_1, domain_train_2, domain_train_3], axis = 0)
domain_test_a = np.concatenate([domain_test_1, domain_test_2, domain_test_3], axis = 0)
"""