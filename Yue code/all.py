# python all.py $D $method $lr $eps_test $epoch

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
import copy
# import torchattacks

import logging
import time
import random

import os.path

from net import *
import sklearn
# from dataset import load_dataset
from torch import linalg as LA
# from manifolds import load_manifold
from utils import *
import sys
import copy
from scipy.stats import ortho_group

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def to_onehot(y, num_classes):
    y = torch.tensor(y,dtype=torch.long)
    return y

def attack_fgsm(model, X, y, epsilon,trim,method='2'):
    # print(method)
    model.eval()
    attack_iters = 1
    alpha = epsilon / attack_iters * 3
    delta = torch.zeros_like(X, requires_grad=True)

    for iter_ in range(attack_iters):
        output = model(X + delta)
        batch_size=X.shape[0]
        channels=len(X.shape)-1
        eps_for_division=1e-10
        shape=(batch_size,)+(1,)*channels
        shape2=(-1,)+(1,)*channels
        # print(output)
        y = y.long()
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        if method == '4':
            grad = torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ).cuda())
        if method == '5':
            grad.requires_grad = False
            grad -= torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ).cuda())

        d = delta + alpha *grad/(torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(shape2)+0.0000001)
        d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1)
        factor = epsilon / (d_norms+0.0000001)
        factor = torch.min(factor, torch.ones_like(d_norms))
        d = d * factor.view(shape2)
        
        delta.data = d # used for loss computation
        # manually change this part for on-manifold attack:
    if method == '1':
        delta.detach()
        delta = torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ).cuda())
    if method == '3':
        delta.detach()
        delta.requires_grad = False
        delta -= torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ).cuda())

    model.train()
    # if trim:
    #     delta=clamp(delta,0-X,1-X)
    return delta.detach()


def attack_fgsm(model, X, y, epsilon,trim,method='2'):
    model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    
    alpha = epsilon / 20 * 3
    attack_iters=20
    for _ in range(1):
        delta = torch.zeros_like(X).cuda()
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            if method == '4':
                grad = torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ).cuda())
            if method == '5':
                grad.requires_grad = False
                grad -= torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ).cuda())

            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data = d # used for loss computation
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if method == '1':
        max_delta.detach()
        max_delta = torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ).cuda())
    if method == '3':
        max_delta.detach()
        max_delta.requires_grad = False
        max_delta -= torch.matmul( max_delta,torch.tensor( m_proj,dtype=torch.float ).cuda())

    model.train()
    return max_delta

def pgd_robustness(model,train_loader,attack='none',epsilon=0.3,LOSS='ce',method='1'):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    
    trim=False
    alpha=1
    attack_iters=1
    restarts=1
    if LOSS=='mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        X=X.float()
        if attack=='none':
            delta=torch.zeros_like(X)
        elif attack=='l2':
            if LOSS=='mse':
                delta = attack_fgsm(model, X, y, epsilon,trim,method)
            else:
                delta = attack_fgsm(model, X, y, epsilon, trim,method)
        elif attack=='linf':
            if LOSS=='mse':
                delta = attack_pgd_linf_mse(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
            else:
                delta = attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
        output = model(X+delta)
        loss = criterion(output, y)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_n += y.size(0)
    return test_loss/test_n, test_acc/test_n



def train(lr,epochs,lr_type='flat',attack='none',epsilon=0.3,LOSS='ce'):
    xdim=x.shape[1]
    if LOSS=='mse':
        model=two_layer_relu_single_output_mse(xdim,'relu',20000).cuda()
    else:
        model=two_layer_relu_single_output_mse(xdim,'relu',20000, n_class).cuda()


    model_init = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    fname='manifold_model/circle_codim-'+str(codim)+'nclass-'+str(n_class)+'.pth'
    Loss,acc=0,0
    # pgd_robustness(model,train_loader, attack='l2',epsilon=eps_test,method='1')
    # print(Loss,acc)
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(train_loader):
            # print(torch.norm(model.first.weight.grad,2).data)
            
            X, y = X.cuda(), y.cuda()
            # print(torch.std(X[:,0]))
            if xdim==1:
                X=X.unsqueeze(1)
            X=X.float()
            # model.first.weight.requires_grad = False
            
            X_adv=X
            # model.first.weight.requires_grad = True

            output = model(X_adv)

            loss = criterion(output, y)

            loss.backward()
            opt.step()
            opt.zero_grad()     
            if LOSS=='mse':
                z=(output>0.5).long()
                train_acc += (z.squeeze() == y).sum().item()
            else:
                train_acc += (output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

        # Loss,acc=pgd_robustness(model,train_loader, attack='l2',epsilon=eps_test,method='1')
        Loss, acc = 0,0
        train_time = time.time()

        if epoch % 100 == 0 or epoch <= 100:
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n,Loss, acc)
            model.eval()
            # print(delta)
        
        if epoch >= epochs-1:
        # if (epoch +1 )% 1000 == 0:
            delta = attack_fgsm(model, X, y, eps_test, False)
            # print(delta)
            delta = delta.cpu()
            delta_ = torch.matmul(delta, torch.tensor(m_proj,dtype=torch.float))
            print(sum(delta_[0]**2)/(eps_test**2),sum(delta_[0]**2)/sum(delta[0]**2) )
            model1 = copy.deepcopy(model)
            model.eval()
            # for eps in np.array([i/10 for i in range(10)]+
            #                     [i/10+1 for i in range(10)]+
            #                     [i/10+2 for i in range(10)]+
            #                     [i/10+3 for i in range(10)]+
            #                     [i/1+2 for i in range(10)]+[100]):
            for eps in np.array(
                            
                            [i/100+0.4 for i in range(10)]+
                            [i/100+0.5 for i in range(10)]+
                            [i/10+0.4 for i in range(20)]
                            )*((codim/D)**0.5):
                            # [i/10+2 for i in range(10)]+
                            # [i/1+2 for i in range(10)]):
                print(eps)
                # eps = eps/10
                # eps = eps*1000
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='1')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='2')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='3')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='4')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='5')

            
            model = model1
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        model.train()
    
    return model

def test(model, attack, epsilon, LOSS,print_=False,method='1'):
    # print(epsilon,method)
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    criterion = nn.CrossEntropyLoss()
 
    for i, (X, y) in enumerate(test_loader):
        
        X, y = X.cuda(), y.cuda()
        X=X.float()
        X_adv=X

        output = model(X_adv)
        loss = criterion(output,y)
        if LOSS=='mse':
            z=(output>0.5).long()
            train_acc += (z.squeeze() == y).sum().item()
        else:
            train_acc += (output.max(1)[1] == y).sum().item()
        train_loss += loss.item() * y.size(0)
        train_n += y.size(0)

    Loss,acc=pgd_robustness(model,test_loader, attack='l2',epsilon=epsilon,method=method)
    train_time = time.time()

    if print_==True:
        tmp = delta.data.cpu().numpy()[:100,]
        print(train_loss/train_n)
        
    logger.info('test \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            train_time - start_time, 0, train_loss/train_n, train_acc/train_n,Loss, acc)

#codim_seq=np.arange(0,510,50)
loss_lst=[]
acc_lst=[]
n_class=2
samples=10
# samples=10000
dummy_scalar=1
std_dev=1
method=(sys.argv[2]) # '1': on manifold, '2': off manifold
epsilon=0.0
noise=1
eps_test=float(sys.argv[4])
D=int(sys.argv[1])
k=10
codim=50
# m = ortho_group.rvs(dim=D)
m = np.random.randn(codim,D)/np.sqrt(D)

m_proj = np.linalg.inv(np.matmul(m, m.transpose()) )
m_proj = np.matmul(m.transpose(),m_proj)
m_proj = np.matmul(m_proj, m)

m_res = np.eye(D) - m_proj


#codim=0
# Create gaussian clusters
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
beta = np.ones(codim)/np.sqrt(codim)

x=np.random.randn(samples*k,codim)*std_dev/np.sqrt(codim)
y=np.random.randn(samples*k)
mu=[]
sign = []
for i in range(k):
    mu.append( np.random.randn(codim) )
    sign.append( k % 2)
    start = int((i-1)*samples)
    end = int(i*samples)
    for j in range(start,end):
        x[j,:] += mu[i] 
        y[j] = sign[i]

shift = np.random.randn(D)/np.sqrt(D)

x= np.matmul(x, m[:codim,:])
for i in range(len(x[:,0])):
    x[i,] += shift

y = to_onehot(y, 2)

x_test=np.random.randn(1000,codim)*std_dev/np.sqrt(codim)
y_test=np.random.rand(1000) 
for i in range(k):
    start = int((i-1)*(1000/k))
    end = int(i*(1000/k))
    # print(start,end)
    for j in range(start,end):
        x_test[j,:] += mu[i] 
        y_test[j] = sign[i]    

x_test= np.matmul(x_test, m[:codim,:])

for i in range(len(x_test[:,0])):
    x_test[i,] += shift

y_test = to_onehot(y_test, 3)

print(codim)

train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
test_data=torch.utils.data.TensorDataset(torch.from_numpy(x_test),y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)


lr = float(sys.argv[3])
epochs = int(sys.argv[5])
train(lr,epochs,attack='l2',epsilon=epsilon)
