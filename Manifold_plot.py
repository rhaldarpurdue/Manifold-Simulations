# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:20:23 2023

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
import matplotlib as mpl
import matplotlib.cm as cm
import torchattacks

import logging
import time

import os.path

from net import *
from sklearn import datasets
from dataset import load_dataset
from torch import linalg as LA
from manifolds import load_manifold

dataset = 'circle784D'#circle3D
input_path='Data/'+dataset
label_path='Data/'+dataset
n_classes=5
xdim=784#784
covariate, response, train_data, test_data=load_manifold(dataset, input_path, label_path,n_classes)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

model=classifier_sigmoid(xdim,n_classes).cuda()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

model.load_state_dict(torch.load('manifold_model/native_pgd.pth'))
model.load_state_dict(torch.load('manifold_model/native_circle784Drandom.pth'))

def train(lr_max,epochs,lr_type='flat',attack='none',epsilon=0.3):
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')
    
    fname='manifold_model/native_'+dataset+attack+'.pth'
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    
    if attack=='fgsm':
        atk=torchattacks.FFGSM(model, eps=epsilon, alpha=epsilon)
    elif attack=='pgd':
        atk=torchattacks.PGD(model, eps=epsilon, alpha=0.01,steps=10)
    elif attack=='cw':
        atk= torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    elif attack=='trades':
        atk=torchattacks.TPGD(model, eps=epsilon, alpha=0.01, steps=50)
    
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
                    
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if xdim==1:
                X=X.unsqueeze(1)
            X=X.float()
            y=y.type(torch.cuda.LongTensor)
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if attack == 'none':
                X_adv=X
            elif attack=='random':
                X_adv=X+torch.zeros_like(X).uniform_(-epsilon,epsilon)
            else:
                X_adv=atk(X,y)
            output = model(X_adv)
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_acc += (output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        torch.save(model.state_dict(), fname)
        
from mpl_toolkits.mplot3d import Axes3D
%matplotlib widget
%matplotlib qt
%matplotlib auto
%matplotlib inline
import matplotlib.pyplot as plt
prob_dot_scale=40
prob_dot_scale_power=3
true_dot_size=50
step=0.05
x_axis_range = np.arange(-1.2,1.2, step)
y_axis_range = np.arange(-1.2,1.2, step)
z_axis_range = np.arange(-1.2,1.2, step)
xx0, xx1, xx2 = np.meshgrid(x_axis_range, y_axis_range,z_axis_range)
xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel(),xx2.ravel()),axis=1),(-1,3))
with torch.no_grad():
    output=model(torch.tensor(xx).float().cuda())
    yy_hat=output.max(dim=1)[1].detach().to('cpu').numpy()
    yy_prob = torch.nn.functional.softmax(output,dim=1).detach().to('cpu').numpy()
yy_size = np.max(yy_prob, axis=1)
fig = plt.figure(figsize=(12, 12))
#ax=Axes3D(fig)
ax = fig.add_subplot(projection='3d')
#ax.scatter(covariate[:,0],covariate[:,1],covariate[:,2],c=response,s=true_dot_size)
ax.scatter(xx[:,0], xx[:,1],xx[:,2], c=yy_hat, alpha=0.1, s=prob_dot_scale*yy_size**prob_dot_scale_power, linewidths=0,)
ax.scatter(covariate[:,0],covariate[:,1],covariate[:,2], c=response, s=true_dot_size, zorder=3, linewidths=0.7, edgecolor='k')
plt.show()