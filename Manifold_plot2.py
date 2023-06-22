
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def train(lr_max,epochs,lr_type='flat',attack='none',epsilon=0.3):
    xdim=x.shape[1]
    #model=classifier_sigmoid(xdim,n_class).cuda()
    model=two_layer_Relu(xdim,n_class).cuda()
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    fname='manifold_model/circle_codim-'+str(codim)+'nclass-'+str(n_class)+'.pth'
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
                trim=False
                alpha=0.01
                attack_iters=50
                restarts=10
                delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                #delta = attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                X_adv=X+delta
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
        #torch.save(model.state_dict(), fname)
    return model

# concentric circles co dimension experiment
#codim_seq=np.arange(0,510,10)
codim_seq=np.arange(0,510,50)
loss_lst=[]
acc_lst=[]
n_class=2
for codim in codim_seq:
    #codim=0
    # Create concentric circles dataset

    for i in range(int((n_class)/2)):
        if i==0:
            X,Y=sklearn.datasets.make_circles(n_samples=200,noise=0.1, factor= 0.3) 
            x=X
            y=Y
        else:
            X,Y=sklearn.datasets.make_circles(n_samples=200,noise=0, factor= 0.3+i*0.2)
            noise=np.random.normal(scale=0.1,size=X.shape) 
            x=np.vstack((x,(3*i)*X+noise)) 
            y=np.hstack((y,Y+2*i))

    #normalising between 0-1
    #x=x-np.min(x)
    #x=x/np.max(x)

    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    #plt.scatter(x[:,0],x[:,1],c=y)
    x=np.hstack((x,np.zeros((x.shape[0],codim))))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    m=train(1e-3,50)
    #m=train(1e-3,50,attack='pgd',epsilon=0.3)
    Loss,acc=pgd_robustness(m)
    l2_loss1,l2_acc1=pgd_robustness(m,attack='l2',epsilon=0.1)
    l2_loss2,l2_acc2=pgd_robustness(m,attack='l2',epsilon=0.2)
    l2_loss3,l2_acc3=pgd_robustness(m,attack='l2',epsilon=0.3)
    linf_loss1,linf_acc1=pgd_robustness(m,attack='linf',epsilon=0.1)
    linf_loss2,linf_acc2=pgd_robustness(m,attack='linf',epsilon=0.2)
    linf_loss3,linf_acc3=pgd_robustness(m,attack='linf',epsilon=0.3)
    loss_lst.append(np.array((Loss,l2_loss1,l2_loss2,l2_loss3,linf_loss1,linf_loss2,linf_loss3)))
    acc_lst.append(np.array((acc,l2_acc1,l2_acc2,l2_acc3,linf_acc1,linf_acc2,linf_acc3)))

acc_array=np.vstack(acc_lst)
loss_array=np.vstack(loss_lst)
file_name='manifold_stats/2LRcircles-'+'nclass-'+str(n_class)
np.save(file_name+"ACC.npy",acc_array)
np.save(file_name+"LOSS.npy",loss_array)
######
# Native dimensions sphere experiment

codim_seq=np.arange(0,510,10) 
loss_lst=[]
acc_lst=[]
n_class=2
for codim in codim_seq:
    dim=codim+2
    for i in range(n_class):
        ur=2*(i+1)-0.4
        lr=2*i
        u=np.random.uniform(low=0, high=1, size=(400,))
        r=(ur-lr)*u**(1/dim)+lr
        X=np.random.normal(size=(400,dim))
        norm=np.linalg.norm(X,axis=1)
        X=r[:,np.newaxis]*X/norm[:,np.newaxis]
        Y=np.repeat(i,400)
        if i ==0:
            x=X
            y=Y
        else:
            x=np.vstack((x,X))
            y=np.hstack((y,Y))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    m=train(1e-3,50)
    Loss,acc=pgd_robustness(m)
    l2_loss1,l2_acc1=pgd_robustness(m,attack='l2',epsilon=0.1)
    l2_loss2,l2_acc2=pgd_robustness(m,attack='l2',epsilon=0.2)
    l2_loss3,l2_acc3=pgd_robustness(m,attack='l2',epsilon=0.3)
    linf_loss1,linf_acc1=pgd_robustness(m,attack='linf',epsilon=0.1)
    linf_loss2,linf_acc2=pgd_robustness(m,attack='linf',epsilon=0.2)
    linf_loss3,linf_acc3=pgd_robustness(m,attack='linf',epsilon=0.3)
    loss_lst.append(np.array((Loss,l2_loss1,l2_loss2,l2_loss3,linf_loss1,linf_loss2,linf_loss3)))
    acc_lst.append(np.array((acc,l2_acc1,l2_acc2,l2_acc3,linf_acc1,linf_acc2,linf_acc3)))

acc_array=np.vstack(acc_lst)
loss_array=np.vstack(loss_lst)
file_name='manifold_stats/2LRspheres-'+'nclass-'+str(n_class)
np.save(file_name+"ACC.npy",acc_array)
np.save(file_name+"LOSS.npy",loss_array)

######
# Native dimensions cubes experiment

codim_seq=np.arange(0,510,10) 
loss_lst=[]
acc_lst=[]
n_class=2
sample=800
for codim in codim_seq:
    dim=codim+2
    for i in range(n_class):
        ur=2*(i+1)-0.4
        lr=2*i
        X=np.random.uniform(low=-ur, high=ur, size=(sample,dim))
        ids=np.max(abs(X),axis=1)>=lr
        X=X[ids]
        while(X.shape[0]<sample):
            temp=np.random.uniform(low=-ur, high=ur, size=(sample,dim))
            ids=np.max(abs(temp),axis=1)>=lr
            temp=temp[ids]
            X=np.vstack((X,temp))
        
        Y=np.repeat(i,X.shape[0])
        if i ==0:
            x=X
            y=Y
        else:
            x=np.vstack((x,X))
            y=np.hstack((y,Y))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    m=train(1e-3,50)
    Loss,acc=pgd_robustness(m)
    l2_loss1,l2_acc1=pgd_robustness(m,attack='l2',epsilon=0.1)
    l2_loss2,l2_acc2=pgd_robustness(m,attack='l2',epsilon=0.2)
    l2_loss3,l2_acc3=pgd_robustness(m,attack='l2',epsilon=0.3)
    linf_loss1,linf_acc1=pgd_robustness(m,attack='linf',epsilon=0.1)
    linf_loss2,linf_acc2=pgd_robustness(m,attack='linf',epsilon=0.2)
    linf_loss3,linf_acc3=pgd_robustness(m,attack='linf',epsilon=0.3)
    loss_lst.append(np.array((Loss,l2_loss1,l2_loss2,l2_loss3,linf_loss1,linf_loss2,linf_loss3)))
    acc_lst.append(np.array((acc,l2_acc1,l2_acc2,l2_acc3,linf_acc1,linf_acc2,linf_acc3)))

acc_array=np.vstack(acc_lst)
loss_array=np.vstack(loss_lst)
file_name='manifold_stats/2LRcubes-'+'nclass-'+str(n_class)
np.save(file_name+"ACC.npy",acc_array)
np.save(file_name+"LOSS.npy",loss_array)

trim=False
alpha=0.01
attack_iters=50
restarts=10
epsilon=0.2
for x_temp,y_temp in train_loader:
    break

delta = attack_pgd_l2(m, x_temp.cuda().float(), y_temp.type(torch.cuda.LongTensor), epsilon, alpha, attack_iters, restarts,trim)
delta = attack_pgd_linf(m, x_temp.cuda().float(), y_temp.type(torch.cuda.LongTensor), epsilon, alpha, attack_iters, restarts,trim)
out=m(x_temp.cuda().float()+delta)
(out.max(1)[1] != y_temp.type(torch.cuda.LongTensor)).sum().item()


def pgd_robustness(model,attack='none',epsilon=0.3):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    
    trim=False
    alpha=0.01
    attack_iters=50
    restarts=10
    
    criterion = nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        X=X.float()
        y=y.type(torch.cuda.LongTensor)
        if attack=='none':
            delta=torch.zeros_like(X)
        elif attack=='l2':
            delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
        elif attack=='linf':
            delta = attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
        output = model(X+delta)
        loss = criterion(output, y)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_n += y.size(0)
    logger.info('Test Loss \t Test Accuracy \t Attack: \t %s',attack)
    logger.info(' \t %.4f \t %.4f', test_loss/test_n, test_acc/test_n)
    return test_loss/test_n, test_acc/test_n

def SMA_convolve(v, n=3):
  return np.convolve(v, np.ones(n), 'valid') / n

def robustness_plot():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LRcircles-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Robustness, circles $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,acc_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,acc_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(acc_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$(D-k)$')
    plt.ylabel(r'Accuracy')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,1])
    plt.savefig('manifold_stats/2LRConcentric_circles_n='+str(n_class)+'.png',dpi=300)
    
def loss_plot():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LRcircles-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Adversarial Loss, circles $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,loss_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,loss_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    #plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    #plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    #plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n],SMA_convolve(loss_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$(D-k)$')
    plt.ylabel(r'Loss')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,5])
    plt.savefig('manifold_stats/2LRLinfConcentric_circles_n='+str(n_class)+'Loss.png',dpi=300)

def robustness_plot2():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LRspheres-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Robustness, $S^{D-1}(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq+2,acc_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq+2,acc_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$D:k=D$')
    plt.ylabel(r'Accuracy')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,1])
    plt.savefig('manifold_stats/2LRSpheres_n='+str(n_class)+'.png',dpi=300)
    
def loss_plot2():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LRspheres-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Adversarial Loss, $S^{D-1} (C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,loss_array[:,0]+2, color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,1]+2, color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,2]+2, color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,3]+2, color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,loss_array[:,4]+2, color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,5]+2, color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,6]+2, color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$D:k=D$')
    plt.ylabel(r'Loss')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,125])
    plt.savefig('manifold_stats/2LRSpheres_n='+str(n_class)+'Loss.png',dpi=300)
    
def robustness_plot2():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LRcubes-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Robustness, Hypercubes $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq+2,acc_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq+2,acc_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq+2,acc_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(acc_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$D:k=D$')
    plt.ylabel(r'Accuracy')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,1])
    plt.savefig('manifold_stats/2LRCubes_n='+str(n_class)+'.png',dpi=300)
    
def loss_plot2():
    plt.style.use('plot_style.txt')
    file_name='manifold_stats/2LR800samplescubes-'+'nclass-'+str(n_class)
    acc_array=np.load(file_name+"ACC.npy")
    loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Adversarial Loss, Hypercubes  $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,loss_array[:,0]+2, color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,1]+2, color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,2]+2, color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,3]+2, color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,loss_array[:,4]+2, color='orange', label=r'$\ell_{\infty}, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,5]+2, color='cyan', label=r'$\ell_{\infty}, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,6]+2, color='magenta', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,1],window), color='yellow', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,2],window), color='blue', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,3],window), color='red', linestyle=':',linewidth=lwd2,alpha=alp2)
    


    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,4],window), color='orange', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,5],window), color='cyan', linestyle=':',linewidth=lwd2,alpha=alp2)
    plt.plot(codim_seq[window-2:n]+2,SMA_convolve(loss_array[:,6],window), color='magenta', linestyle=':',linewidth=lwd2,alpha=alp2)

    plt.xlabel(r'$D:k=D$')
    plt.ylabel(r'Loss')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    
    plt.xlim([0,500])
    plt.ylim([-0.01,125])
    plt.savefig('manifold_stats/2LRCubes_n='+str(n_class)+'Loss.png',dpi=300)