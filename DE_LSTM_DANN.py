#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:24:02 2020

@author: furukawashouya
"""


import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torchsummary import summary
import torch.nn.functional as F


from keras.utils import np_utils
from sklearn import preprocessing
import math
from matplotlib import pyplot as plt




#data_被験者番号_実験回数
data_1_1 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/1_20131027.mat')

#de_被験者番号_実験回数[映画番号-1](62*200*5)
de_1_1 = []
for i in range(1, 16):
    de_1_1.append(data_1_1['de_LDS' + str(i)])


#感情ラベル
label_data = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/ExtractedFeatures/label.mat')

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


#GRLクラスの作成
from torch.autograd import Function

class ReverseLayerF(Function):
    
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)
    
    def backward(ctx, grad_output):
        output = grad_output.neg()*ctx.alpha
        
        return output, None
    
    
#model定義
class DANNModel(nn.Module):
    
    
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(DANNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('lstm', 
                           nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier("layer1", nn.Linear(64, 3))
        self.class_classifier.add_module('classifier_softmax', nn.Softmax())
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier("layer2", nn.Linear(64, 2))
        self.domain_classifier.add_module('domain_sigmoid', nn.Sigmoid())
        
    def forward(self, X_input, alpha, hidden0 = None):
        feature, (hidden, cell) = self.feature(X_input, hidden0)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output
 

"""
class Exstracter(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(Exstracter, self).__init__()
        
        #self.input_dim =  lstm_input_dim
        #self.hidden_dim = lstm_hidden_dim
        
        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True)
        ##lstm_input_dim = 入力特徴量の数 =Vector_size
        ##input = (Batch_Size * Sequence_Length * Vector_Size)
        #self.dense = nn.Linear(lstm_hidden_dim, target_dim)
        
    def forward(self, X_input, hidden0 = None):
        lstm_out, (hidden, cell) = self.lstm(X_input, hidden0)
        
        #linear_out = self.dense(lstm_out[:, -1, :])
        
        return lstm_out

class Class_classifier(nn.Module):
    def __init__(self):
        super(Class_classifier, self).__init__()
        
        self.dense = nn.Linear(64, 3)
        
    def forward(self, input):
        x = self.dense(input)
        softmax = F.softmax(x)
        
        return softmax
    
    
class Domain_classifier(nn.Moudule):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        
        self.dense = nn.Linear(64, 2)
    
    def forward(self, input):
        x = self.dense(input)
        sigmoid = F.sigmoid(x)
        
        return sigmoid

  """
    
def main():
    
    X_train_1, X_test_1, Y_train_1, Y_test_1, domain_train_1, domain_test_1 = setData_imp(de_1_1, label_data, 50)
   
    
    training_size = X_train_1.shape[0]
    test_size = X_test_1.shape[0]
    epochs_num = 3
    hidden_size = 128
    batch_size = 128
    target_dim = 64
    
    
    
    model = DANNModel(310, hidden_size, target_dim)
    model.double()
    
    #summary(model, (batch_size, time_steps, feature_num))
    
    loss_function = nn.CrossEntropyLoss()
   
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
    
    model.train()#学習モード
    for epoch in range(epochs_num):
        #training
        running_loss = 0.0
        training_accuracy = 0.0
        
        for i in range(int(training_size/batch_size)):
            optimizer.zero_grad()
            
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
            training_accuracy += np.sum(np.abs(class_output.item() - label[i] )< 0.1)
        
        model.eval()
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i*batch_size
            data, label, domain = torch.tensor(X_test_1[offset:offset+batch_size]), torch.tensor(Y_test_1[offset:offset+batch_size], torch.tensor(domain_test_1[offset:offset+batch_size]))
            ouput,_ = model(data, None)
                
            
        plt.plot(running_loss)
        training_accuracy /= training_size
        test_accuracy /= test_size
        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy, test_accuracy))
      
            
if __name__ == '__main__':
    main()
