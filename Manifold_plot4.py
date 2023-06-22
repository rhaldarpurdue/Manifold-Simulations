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
import sklearn
from dataset import load_dataset
from torch import linalg as LA
from manifolds import load_manifold
from utils import *

from functorch import hessian,jacrev
from functorch import make_functional
from functorch import vmap, vjp

import pickle

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def sgd_train(x,n_class,train_loader,lr_max,epochs,optimisation_type='Adam',lr_type='flat',attack='none',epsilon=0.3):
    xdim=x.shape[1]
    model=two_layer_Relu(xdim,n_class).cuda()
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')
    criterion = nn.CrossEntropyLoss()
    if optimisation_type=='Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr_max)#SGD to check NKT dynamics
    #fname='manifold_model/circle_codim-'+str(codim)+'nclass-'+str(n_class)+'.pth'
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    
    w0_norms=[]
    w1_norms=[]
    theta_norms=[]
    accuracies=[]
    losses=[]
    score1=[]
    score2=[]
    adv=[]

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        j=0

        for i, (X, y) in enumerate(train_loader):
            j+=1
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
            if j==1:
                score1.append(LA.norm(output[:,0].detach().cpu()))
                score2.append(LA.norm(output[:,1].detach().cpu()))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_acc += (output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)
            torch.cuda.empty_cache()
        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        w0,w1,theta=parameter_norms(model,x) # respective norms
        if epoch%50==0:
            l,advacc=pgd_robustness(model,attack='linf',epsilon=0.1)
            adv.append(advacc)
        w0_norms.append(w0)
        w1_norms.append(w1)
        theta_norms.append(theta)
        losses.append(train_loss/train_n)
        accuracies.append(train_acc/train_n)
        #torch.save(model.state_dict(), fname)
    return w0_norms,w1_norms,theta_norms,losses,accuracies,score1,score2,model,adv

def parameter_norms(model,X):
    #w0, w1, theta  w0,w1 are the weights of the first layer
    D=X.shape[1]
    i=0
    theta_norm=0
    for p in model.parameters():
        if D>2 and i==0:
            w0_norm=LA.norm(p[:,0:2]).detach().cpu()
            w1_norm=LA.norm(p[:,2:D]).detach().cpu()
            theta_norm=w0_norm**2+w1_norm**2
        elif i==0:
            w0_norm=LA.norm(p[:,0:2]).detach().cpu()
            w1_norm=0
            theta_norm=w0_norm**2+w1_norm**2
        theta_norm+=LA.norm(p).detach().cpu()**2
        i+=1
    theta_norm=theta_norm**0.5
    return w0_norm,w1_norm,theta_norm

def concentric_data(codim=0,n_class=2,dummy_scalar=0):
    for i in range(int((n_class)/2)):
        if i==0:
            X,Y=sklearn.datasets.make_circles(n_samples=400,noise=0.1, factor= 0.3) 
            x=X
            y=Y
        else:
            X,Y=sklearn.datasets.make_circles(n_samples=400,noise=0, factor= 0.3+i*0.2)
            noise=np.random.normal(scale=0.1,size=X.shape) 
            x=np.vstack((x,(3*i)*X+noise)) 
            y=np.hstack((y,Y+2*i))
    x=np.hstack((x,dummy_scalar*np.ones((x.shape[0],codim))))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=400, shuffle=True)
    return x,y,train_loader

for cd in [0,10,50]:
    x,y,train_loader=concentric_data(codim,n_class,dummy_scalar=0)
    w0_norms,w1_norms,theta_norms,losses,accuracies,score1,score2,m,adv=sgd_train(x,n_class,train_loader,1e-3, int(1e3),epsilon=0.1)

