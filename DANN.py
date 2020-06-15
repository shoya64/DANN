#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
from functions import ReverseLayerF


class DANNM(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(DANNM, self).__init__()
        
        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True
                           )
        
        self.dense1 = nn.Linear(lstm_hidden_dim, 3)
    
        self.dense2 = nn.Linear(lstm_hidden_dim, 1)
        
    def forward(self, X_input, alpha, hidden0 = None):
        
        #特徴抽出層
        lstm_out, _ = self.lstm(X_input)
        
        #クラス分類層
        class_classifier = self.dense1(lstm_out[:, -1, :])
        
        #GRL
        reverse_feature = ReverseLayerF.apply(lstm_out[:, -1, :], alpha)
        #ドメイン分類層
        domain_classifier = self.dense2(reverse_feature)
        
        return class_classifier, domain_classifier
 
    
    
class DANN(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(DANN, self).__init__()
        
        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True
                           )
        
        
        self.dense1 = nn.Linear(lstm_hidden_dim, 3)
    
        self.dense2 = nn.Linear(lstm_hidden_dim, 1)
        
    def forward(self, X_input, alpha, hidden0 = None):
        
        #特徴抽出層
        lstm_out, _ = self.lstm(X_input)
        print(lstm_out.shape)
        
        #クラス分類層
        class_classifier = self.dense1(lstm_out)
        
        #GRL
        reverse_feature = ReverseLayerF.apply(lstm_out, alpha)
        #ドメイン分類層
        domain_classifier = self.dense2(reverse_feature)
        
        return class_classifier, domain_classifier
 

class DANN_Bi(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(DANN_Bi, self).__init__()
        
        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True
                           )
        
        self.dropout = nn.Dropout(p = 0.3)
        
        self.dense = nn.Linear(lstm_hidden_dim, 3)
    
        self.dense1 = nn.Linear(lstm_hidden_dim, 2)
        
    def forward(self, X_input, alpha, hidden0 = None):
        
        #特徴抽出層
        lstm_out, _ = self.lstm(X_input)
        print(lstm_out.shape)
        
        out = self.dropout(lstm_out[:, -1, :])
        
        
        #クラス分類層
        class_classifier = self.dense(out)
        print(class_classifier.shape)
        
        #GRL
        reverse_feature = ReverseLayerF.apply(out, alpha)
        
        #ドメイン分類層
        domain_classifier = self.dense1(reverse_feature)
        
        return class_classifier, domain_classifier, out
 

class LSTMD(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(LSTMD, self).__init__()
        
        self.lstm = nn.LSTM(input_size = lstm_input_dim, 
                           hidden_size = lstm_hidden_dim,
                           batch_first = True)
        #self.dropout = nn.Dropout(p = 0.01)
        self.dense1 = nn.Linear(lstm_hidden_dim, 64)
        #self.relu = nn.Sigmoid()
        self.dense2 = nn.Linear(64, target_dim)
    def forward(self, X_input, hidden0 = None):
        
        lstm_out, _ = self.lstm(X_input)
        #print(lstm_out[:, -1, :])
        #drop = self.dropout(lstm_out[:, -1, :])
        linear_out1 = self.dense1(lstm_out[:, -1, :])
        #linear_out2 = self.relu(linear_out1)
        #print(linear_out2)
        linear_out3 = self.dense2(linear_out1)
        #print(linear_out3)

        class_classifier = linear_out3
        
        return class_classifier

