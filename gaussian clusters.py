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

def train(lr_max,epochs,lr_type='flat',attack='none',epsilon=0.3,LOSS='ce'):
    xdim=x.shape[1]
    if LOSS=='mse':
        model=two_layer_relu_single_output_mse(xdim).cuda()
    else:
        #model=classifier_sigmoid(xdim,n_class).cuda()
        #model=two_layer_Relu(xdim,n_class).cuda()
        model=two_layer_relu_single_output(xdim, n_class).cuda()
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')
    if LOSS=='mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    fname='manifold_model/circle_codim-'+str(codim)+'nclass-'+str(n_class)+'.pth'
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        output_absmax=0
        output_absmin=1e3
                    
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
            elif attack=='l2':
                trim=False
                alpha=0.01
                attack_iters=50
                restarts=10
                if LOSS=='mse':
                    delta = attack_pgd_l2_mse(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                else:
                    delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                X_adv=X+delta
            else:
                trim=False
                alpha=0.01
                attack_iters=50
                restarts=10
                if LOSS=='mse':
                    delta = attack_pgd_linf_mse(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                else:
                    delta = attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
                X_adv=X+delta
            output = model(X_adv)
            output_absmax=max(output_absmax,output[:,1].abs().max())
            output_absmin=-max(-output_absmin,-output[:,1].abs().min())
            if LOSS=='mse':
                loss = criterion(output.squeeze(),y.float())
            else:
                loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if LOSS=='mse':
                z=(output>0.5).long()
                train_acc += (z.squeeze() == y).sum().item()
            else:
                train_acc += (output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        print("max output:",output_absmax.detach(),"|min output:",output_absmin.detach())
        #torch.save(model.state_dict(), fname)
    return model

# gaussian experiment
codim_seq=np.arange(0,50,10)
#codim_seq=np.arange(0,510,50)
loss_lst=[]
acc_lst=[]
n_class=2
samples=400
dummy_scalar=1
std_dev=0.5
mu1=np.array((2,2))
mu1=mu1.reshape((1,2))
mu2=np.array((-2,2))
mu2=mu2.reshape((1,2))
for codim in codim_seq:
    #codim=0
    # Create gaussian clusters
    x=np.random.randn(samples,2)*std_dev+np.repeat(mu1,samples,0)
    y=np.zeros(samples)
    
    x=np.vstack((x,np.random.randn(samples,2)*std_dev+np.repeat(mu2,samples,0)))
    y=np.hstack((y,np.ones(samples)))

    #plt.scatter(x[:,0],x[:,1],c=y)
    x=np.hstack((x,dummy_scalar*np.ones((x.shape[0],codim))))
    train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    
    
    m=train(1e-3,500)
    #m=train(1e-3,50,attack='pgd',epsilon=0.5)
    Loss,acc=pgd_robustness(m)
    l2_loss1,l2_acc1=pgd_robustness(m,attack='l2',epsilon=0.3)
    l2_loss2,l2_acc2=pgd_robustness(m,attack='l2',epsilon=0.4)
    l2_loss3,l2_acc3=pgd_robustness(m,attack='l2',epsilon=0.5)
    linf_loss1,linf_acc1=pgd_robustness(m,attack='linf',epsilon=0.3)
    linf_loss2,linf_acc2=pgd_robustness(m,attack='linf',epsilon=0.4)
    linf_loss3,linf_acc3=pgd_robustness(m,attack='linf',epsilon=0.5)
    
    '''
    m=train(1e-3,50,LOSS='mse')
    #m=train(1e-3,50,attack='pgd',epsilon=0.3)
    Loss,acc=pgd_robustness(m,LOSS='mse')
    l2_loss1,l2_acc1=pgd_robustness(m,attack='l2',epsilon=0.3,LOSS='mse')
    l2_loss2,l2_acc2=pgd_robustness(m,attack='l2',epsilon=0.4,LOSS='mse')
    l2_loss3,l2_acc3=pgd_robustness(m,attack='l2',epsilon=0.5,LOSS='mse')
    linf_loss1,linf_acc1=pgd_robustness(m,attack='linf',epsilon=0.3,LOSS='mse')
    linf_loss2,linf_acc2=pgd_robustness(m,attack='linf',epsilon=0.4,LOSS='mse')
    linf_loss3,linf_acc3=pgd_robustness(m,attack='linf',epsilon=0.5,LOSS='mse')
    '''
    loss_lst.append(np.array((Loss,l2_loss1,l2_loss2,l2_loss3,linf_loss1,linf_loss2,linf_loss3)))
    acc_lst.append(np.array((acc,l2_acc1,l2_acc2,l2_acc3,linf_acc1,linf_acc2,linf_acc3)))

acc_array=np.vstack(acc_lst)
loss_array=np.vstack(loss_lst)


def pgd_robustness(model,attack='none',epsilon=0.3,LOSS='ce',attack_iters=50):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    
    trim=False
    alpha=0.01
    restarts=10
    
    if LOSS=='mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        X=X.float()
        y=y.type(torch.cuda.LongTensor)
        if attack=='none':
            delta=torch.zeros_like(X)
        elif attack=='l2':
            if LOSS=='mse':
                delta = attack_pgd_l2_mse(model, X, y.float(), epsilon, alpha, attack_iters, restarts,trim)
            else:
                delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts,trim)

        elif attack=='linf':
            if LOSS=='mse':
                delta = attack_pgd_linf_mse(model, X, y.float(), epsilon, alpha, attack_iters, restarts,trim)
            else:
                delta = attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
        output = model(X+delta)
        if LOSS=='mse':
            loss = criterion(output.squeeze(),y.float())
            z=(output>0.5).long()
            test_acc += (z.squeeze() == y).sum().item()
        else:
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
    #file_name='manifold_stats/2LRcircles-'+'nclass-'+str(n_class)
    #acc_array=np.load(file_name+"ACC.npy")
    #loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,50)
    #codim_seq=np.arange(0,510,10)
    n=len(codim_seq)-1
    plt.title(r'Gaussian cluster $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,acc_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.4$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.5$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,acc_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.4$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,acc_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.5$',linewidth=lwd,alpha=alp)
    
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
    #plt.ylim([-0.01,1])
    #plt.savefig('manifold_stats/2LRConcentric_circles_n='+str(n_class)+'.png',dpi=300)
    
def loss_plot():
    plt.style.use('plot_style.txt')
    #file_name='manifold_stats/2LRcircles-'+'nclass-'+str(n_class)
    #acc_array=np.load(file_name+"ACC.npy")
    #loss_array=np.load(file_name+"LOSS.npy")
    window=10
    lwd=0.8
    lwd2=1.5
    alp=0.9
    alp2=0.7
    codim_seq=np.arange(0,510,50)
    n=len(codim_seq)-1
    plt.title(r'Adversarial Loss, Gaussian $(C=$'+str(n_class)+r'$)$')
    plt.plot(codim_seq,loss_array[:,0], color='green', label=r'$\epsilon=0$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,1], color='yellow', label=r'$\ell_2, \epsilon=0.1$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,2], color='blue', label=r'$\ell_2, \epsilon=0.2$',linewidth=lwd,alpha=alp)
    #plt.plot(codim_seq,loss_array[:,3], color='red', label=r'$\ell_2, \epsilon=0.3$',linewidth=lwd,alpha=alp)


    plt.plot(codim_seq,loss_array[:,4], color='orange', label=r'$\ell_{\infty}, \epsilon=0.3$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,5], color='cyan', label=r'$\ell_{\infty}, \epsilon=0.4$',linewidth=lwd,alpha=alp)
    plt.plot(codim_seq,loss_array[:,6], color='magenta', label=r'$\ell_{\infty}, \epsilon=0.5$',linewidth=lwd,alpha=alp)
    
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
    #plt.ylim([-0.01,5])
    #plt.savefig('manifold_stats/2LRLinfConcentric_circles_n='+str(n_class)+'Loss.png',dpi=300)
