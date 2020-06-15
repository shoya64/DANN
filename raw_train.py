#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import scipy.io
#data_被験者番号_実験回数
data_1_1 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/Preprocessed_EEG/1_20131027.mat')


#de_被験者番号_実験回数[映画番号-1](62*200*5)
djc_1_1 = []
for i in range(1, 16):
    djc_1_1.append(data_1_1['djc_eeg' + str(i)])
    

#感情ラベル
label_data = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/Preprocessed_EEG/label.mat')

import torch.nn as nn
import torch.optim as optim
import numpy as np
#import torch.nn.functional as F

from DANN import DANNModel
from DANN import DANNM

from matplotlib import pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


"""
#データ作成
def window_slice(data, time_steps):#dataをtimestepsの長さずつに一つずつデータをずらして小分けする
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    xs = []
    for i in range(data.shape[0] - time_steps + 1):
        xs.append(data[i: i + time_steps])
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs


def window_slice(data, time_steps):#dataをtimestepsの長さずつに一つずつデータをずらして小分けする
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    xs = []
    for i in range(data.shape[0] - time_steps + 1):
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
        if i < 12:
            data_1.append(data[i])
        else :
            data_2.append(data[i])
    Y_train, Y_test = train_test_split(label.reshape(15,), test_size = 3, shuffle = False)
    for data, label in zip(data_1, list(Y_train)):
        X_train.append(window_slice(data, time_steps))
        y_train.extend([label] * len(X_train[-1]))
        
    for data, label in zip(data_2, list(Y_test)):
        X_test.append(window_slice(data, time_steps))
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
    
    idx = np.random.randint(0, train_x.shape[0] - 1)
    batch_x = train_x[idx:batch_size+idx, :, :]
    batch_t = train_t[idx:batch_size+idx]
    batch_domain = domain[idx:batch_size + idx]
    
    return torch.tensor(batch_x), torch.tensor(batch_t), torch.tensor(batch_domain)

X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
   

def main():
    X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
   
    training_size = X_train_1.shape[0]
    test_size = X_test_1.shape[0]
    epochs_num = 3
    hidden_size = 128
    batch_size = 128
    target_dim = 3
    
   
    
    model = DANNM(310, hidden_size, target_dim)
    model.double()
    
    loss_function = nn.CrossEntropyLoss()
   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
    
    model.train()#学習モード

    pltloss = np.array([0])
    e = np.arange(1, epochs_num+1)
    for epoch in range(epochs_num):
        #training
        running_loss = 0.0
        training_accuracy = 0.0
        
        m = 0
        
        for i in range(int(training_size/batch_size)):
            optimizer.zero_grad()
            
            
            #怪しいalpha
            p = float(i + epoch * test_size) / epochs_num / test_size
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            data, label, domain = mkRandomBatch(X_train_1, Y_train_1, domain_train_1, batch_size)
            
            class_output, domain_output = model(data, alpha)
            
            
            class_loss = loss_function(class_output, label)
            domain_loss = loss_function(domain_output, domain)
        
            data_t, label_t, domain_t = mkRandomBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
            _, domain_t_output = model(data_t, alpha)
            
            domain_t_loss = loss_function(domain_t_output, domain_t)
        
            
            err = class_loss + domain_loss + domain_t_loss
            
            err.backward()
            optimizer.step()
            
            
            running_loss += err.item()
            for j in range(class_output.shape[0]):
                m += 1
                if torch.argmax(class_output[j, :]) == label[j]:
                    training_accuracy += 1
                #training_accuracy += torch.sum(torch.abs(class_output[j, :] - label[j] )< 0.1)
            
            
        model.eval()
        test_accuracy = 0.0 
        k = 0
        for i in range(int(test_size/batch_size)):
            
             p = float(i + epoch * test_size) / epochs_num / test_size
             alpha = 2. / (1. + np.exp(-10 * p)) - 1
             data, label, domain = mkRandomBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
             class_t_output ,domain_t_output = model(data, alpha)
             
        for j in range(class_t_output.shape[0]):
            k += 1
            if torch.argmax(class_output[j, :]) == label[j]:
                test_accuracy += 1
        
        if epoch == 0:
            pltloss = np.array([running_loss])
        else:
            pltloss = np.append(pltloss, running_loss)
        
        training_accuracy /= m
        test_accuracy /= k 
        
        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy, test_accuracy))
   
    plt.plot(e, pltloss)   
            
if __name__ == '__main__':
    main()


"""
