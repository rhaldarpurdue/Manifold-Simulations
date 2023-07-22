# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:58:46 2022

@author: rajde
"""
import torch
import torch.nn as nn
# from torch import linalg as LA
import numpy as np

""" utility functions"""
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]
    

def cut(y,h,n_classes):
    filt=torch.nn.functional.one_hot(y,n_classes)
    filt=torch.repeat_interleave(filt,h,1)
    return filt
    
"""Neural Networks"""

    
class two_layer_relu_single_output_mse(nn.Module):
    def __init__(self,xdim,act,h,out=1, norm=1):
        super().__init__()
        # h=1
        self.first = nn.Linear(xdim, h, bias=True)
        if act == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Sigmoid()
        self.second = nn.Linear(h, out,bias=False)
        self.act = act

        if norm != 1:
            self.first.weight.data = self.first.weight.data/torch.std(self.first.weight.data[0,:])/np.sqrt(xdim)*norm
            self.second.weight.data = self.second.weight.data/torch.std(self.second.weight.data)/np.sqrt(h)*norm
        # print(torch.norm(self.first.weight.data,2),torch.std(self.first.weight.data))
        # if h>1:
        #     self.first.weight.data = self.first.weight.data/torch.std(self.first.weight.data[0,:])/np.sqrt(xdim)
        #     self.first.weight.data[int(h/2):,:] = self.first.weight.data[:int(h/2),:]
        #     self.second.weight.data[0,int(h/2):] = -self.second.weight.data[0,:int(h/2)]
        #     self.second.weight.data = self.second.weight.data/torch.std(self.second.weight.data)/np.sqrt(h)    
        # else:
        #     self.second.weight.data = self.second.weight.data/self.second.weight.data

        # print(torch.norm(self.first.weight.data,2),torch.std(self.first.weight.data[:,1]))
        print(torch.std(self.first.weight.data[0,:]))
        # print(torch.norm(self.second.weight.data,2))

    def forward(self, x):
        X = self.first(x)
        if self.act != 'linear':
            if self.act == 'relu':
                X = self.activation(X)
            else:
                X = (self.activation(X)-0.5)*2
        X = self.second(X)
        return X

def simple_classifier(n_classes):
    model = nn.Sequential(nn.Linear(2, 16),
    nn.LeakyReLU(0.01),
    nn.Linear(16,8),
    nn.LeakyReLU(0.01),
    nn.Linear(8,n_classes))
    return model

def simple_classifier_sigmoid(xdim,n_classes):
    model = nn.Sequential(nn.Linear(xdim, 50),
    nn.Sigmoid(),
    nn.Linear(50,100),
    nn.Sigmoid(),
    nn.Linear(100,n_classes))
    return model

def classifier_sigmoid(xdim,n_classes):
    model = nn.Sequential(
        nn.Linear(xdim, 50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,100),
        nn.LeakyReLU(0.01),
        nn.Linear(100,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,n_classes)
        #nn.Sigmoid()
    
    )
    return model

class Attack_net(nn.Module):
    def __init__(self,epsilon, n_classes):
        super().__init__()
        self.C=n_classes
        self.epsilon=epsilon
        self.attack= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100),
            nn.LeakyReLU(0.01),
            nn.Linear(100,15),
            nn.LeakyReLU(0.01),
            nn.Linear(15,2)
            #nn.Sigmoid()
        
        )
    def forward(self, x):
        x_shift = torch.tanh((self.attack(x)))*self.epsilon
        return x_shift

class Attack_net_withlabels0(nn.Module):
    def __init__(self,epsilon,n_classes):
        super().__init__()
        self.C=n_classes
        self.epsilon=epsilon
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100*self.C)
        )
        self.dec= nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100*self.C,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2))
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=(x_shift)*cut(y,h=100,n_classes=self.C) #n_classes=2
        #z=self.dec(x_shift)
        x_shift= torch.tanh(self.dec(x_shift))*self.epsilon
        return x_shift

class Attack_net_withlabelsL2(nn.Module):
    def __init__(self,epsilon,n_classes):
        super().__init__()
        self.C=n_classes
        self.epsilon=epsilon
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100)
        )
        self.decoders=nn.ModuleList([nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2)) for i in range(self.C)])#create n_classes number of decoders
       
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=torch.cat([self.decoders[s](x_shift[i, :]).unsqueeze(0) for i, s in enumerate(y)], axis=0)
        #x_shift= torch.tanh(x_shift)*self.epsilon
        x_norm=LA.vector_norm(x_shift,dim=1)
        x_shift=self.epsilon*x_shift/x_norm.unsqueeze(1)

        return x_shift
        
class Attack_net_withlabelsLINF(nn.Module):
    def __init__(self,epsilon,n_classes):
        super().__init__()
        self.C=n_classes
        self.epsilon=epsilon
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100)
        )
        self.decoders=nn.ModuleList([nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2)) for i in range(self.C)])#create n_classes number of decoders
       
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=torch.cat([self.decoders[s](x_shift[i, :]).unsqueeze(0) for i, s in enumerate(y)], axis=0)
        #x_shift= torch.tanh(x_shift)*self.epsilon
        x_norm=LA.vector_norm(x_shift,ord=float('inf'),dim=1)
        x_shift=self.epsilon*x_shift/x_norm.unsqueeze(1)
        return x_shift

class Attack_net_withlabelsLP(nn.Module):
    def __init__(self,epsilon,n_classes,p=2):
        super().__init__()
        self.p=p
        self.C=n_classes
        self.epsilon=epsilon
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100)
        )
        self.decoders=nn.ModuleList([nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2)) for i in range(self.C)])#create n_classes number of decoders
       
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=torch.cat([self.decoders[s](x_shift[i, :]).unsqueeze(0) for i, s in enumerate(y)], axis=0)
        #x_shift= torch.tanh(x_shift)*self.epsilon
        x_norm=LA.vector_norm(x_shift,ord=self.p,dim=1)
        x_shift=self.epsilon*x_shift/x_norm.unsqueeze(1)
        return x_shift
class Attack_net_withlabelsLP_clamped(nn.Module):
    def __init__(self,epsilon,n_classes,p=2,xdim=2):
        super().__init__()
        self.d=xdim
        self.p=p
        self.C=n_classes
        self.epsilon=epsilon
        self.scale=self.d**(1/self.p)
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100)
        )
        self.decoders=nn.ModuleList([nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2)) for i in range(self.C)])#create n_classes number of decoders
       
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=torch.cat([self.decoders[s](x_shift[i, :]).unsqueeze(0) for i, s in enumerate(y)], axis=0)
        #x_shift= torch.tanh(x_shift)*self.epsilon
        x_norm=LA.vector_norm(x_shift,ord=self.p,dim=1)
        x_shift=(self.scale)*self.epsilon*x_shift/x_norm.unsqueeze(1)
        x_shift=torch.clamp(x_shift,-self.epsilon,self.epsilon)
        return x_shift


class regressor(nn.Module):
    def __init__(self,xdim=13):
        super().__init__()
        self.d=xdim
        self.net=nn.Sequential(nn.Linear(xdim,50),
                               nn.LeakyReLU(0.01),
                               nn.Linear(50,20),
                               nn.LeakyReLU(0.01),
                               nn.Linear(20,1))
    def forward(self,x):
        return self.net(x)
class lp_regression_atk(nn.Module):
    def __init__(self,epsilon,p=2,xdim=13):
        super().__init__()
        self.p=p
        self.epsilon=epsilon
        self.d=xdim
        self.scale=self.d**(1/self.p)
        self.net=nn.Sequential(nn.Linear(xdim,50),
                               nn.LeakyReLU(0.01),
                               nn.Linear(50,50),
                               nn.LeakyReLU(0.01),
                               nn.Linear(50,xdim))
        self.magnitude=nn.Sequential(nn.Linear(xdim,20),
                                     nn.LeakyReLU(0.01),
                                     nn.Linear(20,1),
                                     nn.Sigmoid()
                                     )
    def forward(self,x):
        x_shift = self.net(x)
        x_norm=LA.vector_norm(x_shift,ord=self.p,dim=1)
        x_shift=(self.scale)*self.epsilon*x_shift/x_norm.unsqueeze(1)
        return x_shift*self.magnitude(x)
        
class netG(nn.Module):
    def __init__(self,input_nc, image_nc):
        super().__init__()
        self.attack= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100),
            nn.LeakyReLU(0.01),
            nn.Linear(100,15),
            nn.LeakyReLU(0.01),
            nn.Linear(15,2)
            #nn.Sigmoid()
        
        )
    def forward(self, x):
        return self.attack(x)
    
class netD(nn.Module):
    def __init__(self, image_nc):
        super().__init__()
        self.prob= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100),
            nn.LeakyReLU(0.01),
            nn.Linear(100,15),
            nn.LeakyReLU(0.01),
            nn.Linear(15,1),
            nn.Sigmoid()
        
        )
    def forward(self, x):
        return self.prob(x)
    
class netG_withlabels(nn.Module):
    def __init__(self,input_nc, image_nc,n_classes):
        super().__init__()
        self.C=n_classes
        self.enc= nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(0.01),
            nn.Linear(50,100*self.C)
        )
        self.dec= nn.Sequential(nn.LeakyReLU(0.01),
        nn.Linear(100*self.C,50),
        nn.LeakyReLU(0.01),
        nn.Linear(50,15),
        nn.LeakyReLU(0.01),
        nn.Linear(15,2))
    def forward(self, x, y):
        x_shift = self.enc(x)
        x_shift=(x_shift)*cut(y,h=100,n_classes=self.C) #n_classes=2
        x_shift= self.dec(x_shift)
        return x_shift