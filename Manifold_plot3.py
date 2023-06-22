# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:54:39 2023

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
    model=classifier_sigmoid(xdim,n_class).cuda()
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
    g_fx_theta=[]
    g_fxd_theta=[]
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
            g1,g2=grad_diffs(X, y, model,epsilon)
            g_fx_theta.append(g1)
            g_fxd_theta.append(g2)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)
            torch.cuda.empty_cache()
        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        #torch.save(model.state_dict(), fname)
    return g_fx_theta,g_fxd_theta

def concentric_data(codim=0,n_class=2):
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
    x=np.hstack((x,np.zeros((x.shape[0],codim))))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=400, shuffle=True)
    return x,y,train_loader


def grad_diffs(X,Y,model,epsilon=0.3):
    trim=False
    alpha=0.01
    attack_iters=50
    restarts=10
    delta = attack_pgd_linf(model, X.cuda().float(), Y.type(torch.cuda.LongTensor), epsilon, alpha, attack_iters, restarts,trim)
    functional_model, params = make_functional(model,disable_autograd_tracking=True) #f(x,theta_t) theta_t
    def functional_loss(params, inputs, labels):
        outputs = functional_model(params, inputs)
        loss = F.cross_entropy(outputs, labels,reduction='sum')
        return loss
    #functional_model(params,X) # f(x,theta_t)
    #compute_batch_jacobian = vmap(jacrev(functional_model, argnums=0), in_dims=(None, 0)) # in dims  batch dimensions of (params,input)
    compute_batch_jacobian = vmap(jacrev(functional_loss, argnums=0), in_dims=(None, 0,0))
    #grad_f_theta_x = compute_batch_jacobian(params, X)  # tuple gradient w.r.t. parameters for each sample
    grad_f_theta_x = compute_batch_jacobian(params, X,Y)
    grad_f_theta_x = tuple(t.cpu() for t in grad_f_theta_x)
    #grad_f_theta_x_prime=compute_batch_jacobian(params, X+delta)
    grad_f_theta_x_prime=compute_batch_jacobian(params, X+delta,Y)
    grad_f_theta_x_prime = tuple(t.cpu() for t in grad_f_theta_x_prime)
    return grad_f_theta_x, grad_f_theta_x_prime

def grad_computations(optimisation_type,codim=0,n_class=2,epsilon=0.3):
    x,y,train_loader=concentric_data(codim,n_class)
    if optimisation_type=='Adam':
        glist,glist2=sgd_train(x,n_class,train_loader,1e-3, 140,epsilon=epsilon)
    else:
        glist,glist2=sgd_train(x,n_class,train_loader,1e-1, 300,optimisation_type='SGD',epsilon=epsilon)
    return glist,glist2


def plot_training_dynamics(glist,glist2,sample_id,w_index):
    plt_lst=[]
    plt_original_lst=[]
    plt_adv=[]
    for g1,g2 in zip(glist,glist2):
        if sample_id=='all':
            plt_lst.append(LA.norm((g1[w_index]-g2[w_index]).flatten()))
            plt_original_lst.append(LA.norm(g1[w_index].flatten()))
            plt_adv.append(LA.norm(g2[w_index].flatten()))
        else:
            plt_lst.append(LA.norm((g1[w_index][sample_id]-g2[w_index][sample_id]).flatten()))
            plt_original_lst.append(LA.norm(g1[w_index][sample_id].flatten()))
            plt_adv.append(LA.norm(g2[w_index][sample_id].flatten()))
    return plt_lst,plt_original_lst,plt_adv

gradients_theta_x=[]
gradients_theta_xplusD=[]
for cd in [0,10,50]:
    glist,glist2=grad_computations(optimisation_type='Adam',codim=cd,n_class=2,epsilon=0.1)
    gradients_theta_x.append(glist)
    gradients_theta_xplusD.append(glist2)

with open('grads.pickle', 'wb') as handle:
    pickle.dump(gradients_theta_x, handle)
with open('gradsPerturbed.pickle', 'wb') as handle:
    pickle.dump(gradients_theta_xplusD, handle)

def compare_plot(sample_id='all',w_index=0):
    plt.style.use('plot_style.txt')
    codim0diff,codim0,codim0adv=plot_training_dynamics(gradients_theta_x[0],gradients_theta_xplusD[0],sample_id,w_index)
    codim50diff,codim50,codim50adv=plot_training_dynamics(gradients_theta_x[1],gradients_theta_xplusD[1],sample_id,w_index)
    codim100diff,codim100,codim100adv=plot_training_dynamics(gradients_theta_x[2],gradients_theta_xplusD[2],sample_id,w_index)


    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7

    plt.title(r'Gradient norm w.r.t. time, circles $(C=2)$ '+str(w_index)+r'group $\theta;\ell_{\infty},\epsilon=0.1$')
    plt.plot(codim0diff, color='green', label=r'$(D-k)=0,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    plt.plot(codim0, color='green', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    plt.plot(codim0adv, color='green', label=r'$\nabla_{\theta_t}L(\theta_t,x+\delta)$',linewidth=lwd2,alpha=alp2,linestyle='--')
    
    plt.plot(codim50diff, color='red', label=r'$(D-k)=50,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    plt.plot(codim50, color='red', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    plt.plot(codim50adv, color='red', label=r'$||\nabla_{\theta_t}L(\theta_t,x+\delta)||$',linewidth=lwd2,alpha=alp2,linestyle='--')
    
    plt.plot(codim100diff,color='blue', label=r'$(D-k)=100,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    plt.plot(codim100,color='blue', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    plt.plot(codim100adv,color='blue', label=r'$||\nabla_{\theta_t}L(\theta_t,x+\delta)||$',linewidth=lwd2,alpha=alp2,linestyle='--')

    plt.xlabel(r'$t$')
    plt.ylabel(r'$||\nabla_{\theta_t}\mathcal{L}||$')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,140])
    plt.ylim([-0.01,200])
    #plt.savefig('manifold_stats/Concentric_circles_nclass=2_training_dynamic_sample'+str(sample_id)+'Wts'+str(w_index)+'.png',dpi=300)




def compare_plot(sample_id='all',w_index=0):
    plt.style.use('plot_style.txt')
    codim0diff,codim0,codim0adv=plot_training_dynamics(gradients_theta_x[0],gradients_theta_xplusD[0],sample_id,w_index)
    codim50diff,codim50,codim50adv=plot_training_dynamics(gradients_theta_x[1],gradients_theta_xplusD[1],sample_id,w_index)
    codim100diff,codim100,codim100adv=plot_training_dynamics(gradients_theta_x[2],gradients_theta_xplusD[2],sample_id,w_index)


    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.set_ylabel(r'$||\nabla_{\theta_t}\mathcal{L}(x)||$')
    ax2.set_ylabel(r'$||\nabla_{\theta_t}\mathcal{L}(x+\delta)||$')


    fig.suptitle(r'Gradient norm w.r.t. time, circles $(C=2)$ '+str(w_index)+r'group $\theta;\ell_{\infty},\epsilon=0.1$')
    ax2.plot(codim0diff, color='green', label=r'$(D-k)=0,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    ax1.plot(codim0, color='green', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    ax2.plot(codim0adv, color='green', label=r'$\nabla_{\theta_t}L(\theta_t,x+\delta)$',linewidth=lwd2,alpha=alp2,linestyle='--')
    
    ax2.plot(codim50diff, color='red', label=r'$(D-k)=10,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    ax1.plot(codim50, color='red', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    ax2.plot(codim50adv, color='red', label=r'$||\nabla_{\theta_t}L(\theta_t,x+\delta)||$',linewidth=lwd2,alpha=alp2,linestyle='--')
    
    ax2.plot(codim100diff,color='blue', label=r'$(D-k)=50,||\nabla_{\theta_t}L(\theta_t,x+\delta)-\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp)
    ax1.plot(codim100,color='blue', label=r'$||\nabla_{\theta_t}L(\theta_t,x)||$',linewidth=lwd,alpha=alp,linestyle=':')
    ax2.plot(codim100adv,color='blue', label=r'$||\nabla_{\theta_t}L(\theta_t,x+\delta)||$',linewidth=lwd2,alpha=alp2,linestyle='--')

    ax1.set_xlabel(r'$t$')
    fig.legend(fontsize='x-small',ncol=2,loc='lower right')
    
    ax1.set_xlim([0,140])
    ax1.set_ylim([-0.01,20])
    ax2.set_ylim([-0.01,500])
    fig.savefig('manifold_stats/Concentric_circles_nclass=2_training_dynamic_sample'+str(sample_id)+'Wts'+str(w_index)+'.png',dpi=300)


