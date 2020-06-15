#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:39:09 2020

@author: furukawashouya
"""


import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import torch.nn.functional as F

from DANN import DANNM
from DANN import DANN_Bi

from matplotlib import pyplot as plt

import torch


from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.metrics import confusion_matrix
import seaborn as sns

#data_被験者番号_実験回数
data_1_1 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/1_20131027.mat')

#de_被験者番号_実験回数[映画番号-1](62*200*5)
de_1_1 = []
for i in range(1, 16):
    de_1_1.append(data_1_1['de_LDS' + str(i)])

#感情ラベル
label_data = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/label.mat')

def window_slice_MinMax(data, time_steps):
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    mm = preprocessing.MinMaxScaler()
    data_min_max = mm.fit_transform(data)
    xs = []
    for i in range(data_min_max.shape[0] - time_steps + 1):
        xs.append(data_min_max[i: i + time_steps])
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    return xs


def setData(data, label_data):#data(リスト:長さ15)
    x = [0]*9
    con_data = [0]*9
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    label = label_data['label']
    label = label[0]
    
    mm = preprocessing.MinMaxScaler()
    
    for i in range(len(data)):#リスト型のdataの要素それぞれでデータ整形
        X = data[i] ##de_1_1のデータをXに
        for k in range(62):#62チャンネルで時系列を9個に分割して分割した中央値をとる
            x[0], x[1],x[2], x[3], x[4], x[5], x[6], x[7], x[8] = np.array_split(X[k,:,:], 9, 0)
            for j in range(len(x)):
                median_data = np.median(x[j], axis = 0, keepdims = False)#median_data=(5,)
                if k == 0:
                    con_data[j] = median_data
                else:
                    con_data[j] = np.concatenate([con_data[j], median_data])
                    
        for l in range(9):
            con_data[l] = con_data[l].reshape(310,1)
            if l == 0:
                input_data = con_data[l]
            else:
                input_data = np.hstack([input_data, con_data[l]])#input_data.shape = (310, 9)
                #input_data = np.vstack([input_data, con_data[l]])#input_data.shape(310*9,1)
                
            
        
        #X_train_all.append(input_data.T)
        if i < 9:
            input_data = input_data.T
            L = mm.fit_transform(input_data)
            X_train.append(L)
            
        else:
            input_data = input_data.T
            L = mm.fit_transform(input_data)
            X_test.append(L)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    
    for i in range(label.shape[0]):
        if i < 9:
            y_train.append(label[i])
        else:
            y_test.append(label[i])
     
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    plus1 = np.array([1]*y_train.shape[0])
    plus2 = np.array([1]*y_test.shape[0])
    y_train = y_train + plus1
    y_test = y_test + plus2
    
    #ドメインラベル,1:train,0:test
    domain_train = np.array([1]*y_train.shape[0])
    domain_test = np.array([0]*y_test.shape[0])
    
    return X_train, X_test, y_train, y_test, domain_train, domain_test

def mkRandomBatch(train_x, train_t, domain, batch_size):
    
    idx = np.random.randint(0, train_x.shape[0] - 1)
    batch_x = train_x[idx:batch_size+idx, :, :]
    batch_t = train_t[idx:batch_size+idx]
    batch_domain = domain[idx:batch_size + idx]
    
    return torch.tensor(batch_x), torch.tensor(batch_t), torch.tensor(batch_domain)

X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData(de_1_1, label_data)

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

def main():
    #viz = Visdom()  # インスタンスを作る
    
    X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData(de_1_1, label_data)
   
    training_size = X_train_1.shape[0]
    test_size = X_test_1.shape[0]
    epochs_num = 50
    hidden_size = 3
    batch_size = 1
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
    
    b_x, b_t, d = mkBatch(X_train_1, Y_train_1, domain_train_1, batch_size)
    #b_x_t, b_t_t, d_t = mkBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
    for epoch in range(epochs_num):
        #training
        running_loss = 0.0
        training_accuracy = 0.0
        label_eval = []
        class_eval = []
        running_class_loss = 0.0
        running_domain_loss = 0.0
        
        m = 0
        
        rand = rand_ints_nodup(0, len(b_x)-1, len(b_x))
        #rand_t = rand_ints_nodup(0, len(b_x_t)-1, len(b_x_t))
        
        for i in range(int(training_size/batch_size)):
            optimizer.zero_grad()
            
            
            #GRLのalpha
            p = float(i + epoch * test_size) / epochs_num / test_size
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 0.9
            
            #バッチデータ取得
            data, label, domain = torch.tensor(b_x[rand[i]]), torch.tensor(b_t[rand[i]]), torch.tensor(d[rand[i]])
            
            #モデルに入力
            class_output, domain_output = model(data, alpha)
            
            #ラベル分類の損失
            class_loss = class_loss_function(class_output, label)
            
            #ドメインサイズ成形
            domain = domain.view(domain.size(0), 1)
            
            #ドメイン分類の損失
            domain_loss = domain_loss_function(domain_output, domain.float())
            
            #テストデータの取得
            data_t, label_t, domain_t = mkRandomBatch(X_test_1, Y_test_1, domain_test_1, batch_size)
            
            #テストデータのドメイン分類の出力を取得
            _, domain_t_output = model(data_t, alpha)
            
            domain_t = domain_t.view(domain_t.size(0), 1)
            
            #テストデータのドメインの損失を取得
            domain_t_loss = domain_loss_function(domain_t_output, domain_t.float())
            
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
        alpha = 0.9
        
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
    
    """
    plt.plot(e, pltclass)   
    plt.title('class loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.plot(e, pltdomain)   
    plt.title('domain loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    """
           
    c = confusion_matrix(label_eval, class_eval)
    print(c)
    sns.heatmap(c, annot = True, fmt = 'g', square = True)
    plt.title("training")
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



    
