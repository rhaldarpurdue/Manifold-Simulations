# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:04:37 2023

@author: rajde
"""

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

import logging
import time

import os.path

from net import *

from sklearn import datasets
import math 

def load_manifold(dataset, input_path, label_path,n_classes):
    input_path=input_path+'_'+str(n_classes)+'.npy'
    label_path=label_path+'_'+str(n_classes)+'-label.npy'
    if dataset == 'angles1D':
        if not os.path.isfile(input_path): 
            X=np.random.uniform(0,2*math.pi,1200)
            Y=np.floor(n_classes*X/(2*math.pi))
            #plt.hist(Y,bins=20)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)

        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:1000,]),torch.from_numpy(Y[0:1000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[1000:len(Y),]),torch.from_numpy(Y[1000:len(Y)]))
    elif dataset == 'circle3D':
        if not os.path.isfile(input_path): 
            t=np.random.uniform(0,2*math.pi,1200)
            r=1
            X=np.array(r*(np.cos(t),np.sin(t),0*np.cos(t)))
            X=np.transpose(X)
            Y=np.floor(n_classes*t/(2*math.pi))
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))
    elif dataset == 'circle784D':
        if not os.path.isfile(input_path): 
            t=np.random.uniform(0,2*math.pi,1200)
            r=1
            X=np.array(r*(np.cos(t),np.sin(t),0*np.cos(t)))
            X=np.vstack((X,np.zeros((781,1200))))
            X=np.transpose(X)
            Y=np.floor(n_classes*t/(2*math.pi))
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))

    return X, Y, train_data, test_data