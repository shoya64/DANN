#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:47:44 2020

@author: furukawashouya
"""


import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from sklearn import preprocessing

data_1_1 = scipy.io.loadmat('/Users/furukawashouya/SEED/SEED/Preprocessed_EEG/1_20131027.mat')

