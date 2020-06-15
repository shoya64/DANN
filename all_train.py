#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:27:13 2020

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

X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
X_train_2, X_test_2, Y_train_2, Y_test_2, domain_train_2, domain_test_2 = setData_imp(de_1_2, label_data, 50)
X_train_3, X_test_3, Y_train_3, Y_test_3, domain_train_3, domain_test_3 = setData_imp(de_1_3, label_data, 50)

X_train_a = np.concatenate([X_train_1, X_train_2, X_train_3], axis = 0)
X_test_a = np.concatenate([X_test_1, X_test_2, X_test_3], axis = 0)
Y_train_a = np.concatenate([Y_train_1, Y_train_2, Y_train_3], axis = 0)
Y_test_a = np.concatenate([Y_test_1, Y_test_2, Y_test_3], axis = 0)
domain_train_a = np.concatenate([domain_train_1, domain_train_2, domain_train_3], axis = 0)
domain_test_a = np.concatenate([domain_test_1, domain_test_2, domain_test_3], axis = 0)


def main():
    #viz = Visdom()  # インスタンスを作る
    X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
    X_train_2, X_test_2, Y_train_2, Y_test_2, domain_train_2, domain_test_2 = setData_imp(de_1_2, label_data, 50)
    X_train_3, X_test_3, Y_train_3, Y_test_3, domain_train_3, domain_test_3 = setData_imp(de_1_3, label_data, 50)
    
    X_train_a = np.concatenate([X_train_1, X_train_2, X_train_3], axis = 0)
    X_test_a = np.concatenate([X_test_1, X_test_2, X_test_3], axis = 0)
    Y_train_a = np.concatenate([Y_train_1, Y_train_2, Y_train_3], axis = 0)
    Y_test_a = np.concatenate([Y_test_1, Y_test_2, Y_test_3], axis = 0)
    domain_train_a = np.concatenate([domain_train_1, domain_train_2, domain_train_3], axis = 0)
    domain_test_a = np.concatenate([domain_test_1, domain_test_2, domain_test_3], axis = 0)

    
    training_size = X_train_1.shape[0]
    test_size = X_test_1.shape[0]
    epochs_num = 100
    hidden_size = 32
    batch_size = 32
    target_dim = 3
   
    
    model = DANN_Bi(310, hidden_size, target_dim)
    model.double()
    
    #summary(model, input_size=((100, 50, 310), 1))
    
    #すでにsoftmaxとsigmoidされている
    class_loss_function = nn.CrossEntropyLoss()
    domain_loss_function = nn.BCEWithLogitsLoss()
   
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
    
    
    model.train()#学習モード

    pltloss = np.array([0])
    pltclass = np.array([0])
    pltdomain = np.array([0])
    e = np.arange(1, epochs_num+1)
    
    b_x, b_t, d = mkBatch(X_train_a, Y_train_a, domain_train_a, batch_size)
    #b_x_t, b_t_t, d_t = mkBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
    for epoch in range(epochs_num):
        #training
        running_loss = 0.0
        training_accuracy = 0.0
        label_eval = []
        class_eval = []
        d_label_eval = []
        domain_eval = []
        
        running_class_loss = 0.0
        running_domain_loss = 0.0
        
        m = 0
        
        rand = rand_ints_nodup(0, len(b_x)-1, len(b_x))
        #rand_t = rand_ints_nodup(0, len(b_x_t)-1, len(b_x_t))
        
        for i in range(int(training_size/batch_size)):
            optimizer.zero_grad()
            
            
            #GRLのalpha
            p = float(i + epoch * training_size) / epochs_num / training_size
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 1.0
            
            #バッチデータ取得
            data, label, domain = torch.tensor(b_x[rand[i]]), torch.tensor(b_t[rand[i]]), torch.tensor(d[rand[i]])
            
            #モデルに入力
            class_output, domain_output = model(data, alpha)
            
            #ラベル分類の損失
            class_loss = class_loss_function(class_output, label)
           
            
            #ドメインサイズ成形
            #domain = domain.view(domain.size(0), 1)
           
            #ドメイン分類の損失
            #domain_loss = domain_loss_function(domain_output, domain.float())
            domain_loss = class_loss_function(domain_output, domain)
            
            #テストデータの取得
            data_t, label_t, domain_t = mkRandomBatch(X_test_a, Y_test_a, domain_test_a, batch_size)
            
            #テストデータのドメイン分類の出力を取得
            _, domain_t_output = model(data_t, alpha)
            #print(domain_t_output)
            
            #domain_t = domain_t.view(domain_t.size(0), 1)
            
            #テストデータのドメインの損失を取得
            #domain_t_loss = domain_loss_function(domain_t_output, domain_t.float())
            domain_t_loss = class_loss_function(domain_t_output, domain_t)
            
            #クラスラベル分類の損失，ドレーニングドメインラベル分類の損失，テストドメインラベル分類の損失を一つに
            err = class_loss + domain_loss + domain_t_loss
            
            err.backward()
            optimizer.step()
            
            
            running_loss += err.item()
            running_class_loss += class_loss.item()
            running_domain_loss += domain_loss.item()
            
            
            for j in range(class_output.shape[0]):
                m += 1
                if epoch == epochs_num - 1:
                    label_eval.append(label[j])
                    class_eval.append(torch.argmax(class_output[j, :]))
                    
                if torch.argmax(class_output[j, :]) == label[j]:
                    training_accuracy += 1
                    
            for j in range(domain_output.shape[0]):
                if epoch == epochs_num - 1:
                    d_label_eval.append(domain[j])
                    domain_eval.append(torch.argmax(domain_output[j, :]))
                    #if domain_output[j] > 0:
                     #   domain_eval.append(1)
                    #else:
                     #   domain_eval.append(0)
                        
            for j in range(domain_t_output.shape[0]):
                if epoch == epochs_num - 1:
                    d_label_eval.append(domain_t[j])
                    domain_eval.append(torch.argmax(domain_t_output[j, :]))
                    #if domain_t_output[j] > 0:
                     #   domain_eval.append(1)
                        
                    #else:
                     #   domain_eval.append(0)
                     
            
        
        if epoch == 0:
            pltloss = np.array([running_loss])
            pltclass = np.array([running_class_loss])
            pltdomain = np.array([running_domain_loss])
        else:
            pltloss = np.append(pltloss, running_loss)
            pltclass = np.append(pltclass, running_class_loss)
            pltdomain = np.append(pltdomain, running_domain_loss)
            
            
        #viz.line(X=np.array([epoch]), Y=np.array([running_class_loss]), win='loss', name='class_loss', update='append')
        #viz.line(X=np.array([epoch]), Y=np.array([running_domain_loss]), win='loss', name='domain_acc', update='append')
        
        
        training_accuracy /= m
        
        print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))
   
    
    
    model.eval()
    
    k = 0
    b_x, b_t, d = mkBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
    rand = rand_ints_nodup(0, len(b_x)-1, len(b_x))
    test_accuracy = 0.0
    class_eval_t = []
    label_eval_t = []
    for i in range(int(test_size/batch_size)):
        p = float(i + epoch * test_size) / epochs_num / test_size
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        data, label, domain = torch.tensor(b_x[rand[i]]), torch.tensor(b_t[rand[i]]), torch.tensor(d[rand[i]])
        class_t_output ,domain_t_output = model(data, alpha)
             
        for j in range(class_t_output.shape[0]):
            k += 1
            if epoch == epochs_num - 1:
                label_eval_t.append(label[j])
                class_eval_t.append(torch.argmax(class_t_output[j, :]))
            if torch.argmax(class_t_output[j, :]) == label[j]:
                test_accuracy += 1
        
    test_accuracy /= k 
    
    print('training_accuracy: %.5f' % (test_accuracy))
    
    fig, ax = plt.subplots()
    
    c1, c2, c3 = "blue", "green", "red"
    l1, l2, l3 = "class", "domain", "all"
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Model loss')
    ax.grid()
    
    ax.plot(e, pltclass, color = c1, label = l1)
    ax.plot(e, pltdomain, color = c2, label = l2)
    #ax.plot(e,pltloss, color = c3, label = l3)
    ax.legend(loc = 0)
    fig.tight_layout()
    plt.show()
    

    plt.plot(e, pltloss)   
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
           
    c = confusion_matrix(label_eval, class_eval)
    print(c)
    sns.heatmap(c, annot = True, fmt = 'g', square = True)
    plt.title("training")
    plt.xlabel("predict")
    plt.ylabel("true")
    plt.show()
    
    cd = confusion_matrix(d_label_eval, domain_eval)
    print(cd)
    sns.heatmap(cd, annot = True, fmt = 'g', square = True)
    plt.title("domain_eval")
    plt.xlabel("predict")
    plt.ylabel("true")
    plt.show()

    
    ct = confusion_matrix(label_eval_t, class_eval_t)
    print(ct)
    sns.heatmap(ct, annot = True, fmt = 'g', square = True)
    plt.title("test")
    plt.xlabel("predict")
    plt.ylabel("true")
    plt.show()

    
if __name__ == '__main__':
    main()

    