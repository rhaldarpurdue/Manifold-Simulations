#python change_shift_cpu.py $D $codim $lr $tmp_eps $epoch $seed $h $method $n

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
# import torchattacks
from torch.autograd import Variable
import torch.optim as optim

import logging
import time
import random

import os.path

from net import *
import sklearn
# from dataset import load_dataset
# from torch import linalg as LA
# from manifolds import load_manifold
from utils import *
import sys
import copy
from scipy.stats import ortho_group
from imp import reload

reload(logging)

logger = logging.getLogger(__name__)

h=int(sys.argv[7])

logging.basicConfig(
    # filename='res/change_shift'+sys.argv[1]+'_'+sys.argv[6]+'_linf.txt',
    filename='res/change_shift'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(h)+'_'+sys.argv[8]+'_'+sys.argv[9]+'_'+sys.argv[6]+'_l2.txt',
                    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')

def to_onehot(y, num_classes):
    y = torch.tensor(y,dtype=torch.long)
    return y

def attack_fgsm(model, X, y, epsilon,trim,method='2'):
    model.eval()
    attack_iters = 20
    alpha = epsilon / attack_iters * 3
    delta = torch.zeros_like(X, requires_grad=True)

    for iter_ in range(attack_iters):
        output = model(X + delta)
        batch_size=X.shape[0]
        channels=len(X.shape)-1
        eps_for_division=1e-10
        shape=(batch_size,)+(1,)*channels
        shape2=(-1,)+(1,)*channels
        y = y.long()
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        if method == '4':
            grad = torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ))
        if method == '5':
            grad.requires_grad = False
            grad -= torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ))

        d = delta + alpha *grad/(torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(shape2)+0.0000001)
        d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1).detach()
        factor = epsilon / (d_norms+0.0000001)
        factor = torch.min(factor, torch.ones_like(d_norms))
        d = d * factor.view(shape2)
        
        delta.data = d # used for loss computation
        # manually change this part for on-manifold attack:
    if method == '1':
        delta.detach()
        delta = torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ))
    if method == '3':
        delta.detach()
        delta.requires_grad = False
        delta -= torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ))

    model.train()
    return delta.detach()


def attack_fgsm_linf(model, X, y, epsilon,trim,method='2'):
    model.eval()
    model.first.weight.requires_grad = False
    model.second.weight.requires_grad = False
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    
    alpha = epsilon / 20 * 3
    attack_iters=20
    for _ in range(1):
        delta = torch.zeros_like(X)
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            if method == '4':
                grad = torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ))
            if method == '5':
                grad.requires_grad = False
                grad -= torch.matmul( grad,torch.tensor( m_proj,dtype=torch.float ))

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
        max_delta = torch.matmul( delta,torch.tensor( m_proj,dtype=torch.float ))
    if method == '3':
        max_delta.detach()
        max_delta.requires_grad = False
        max_delta -= torch.matmul( max_delta,torch.tensor( m_proj,dtype=torch.float ))

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
        X, y = X, y
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
        model=two_layer_relu_single_output_mse(xdim,'relu',h)
    else:
        model=two_layer_relu_single_output_mse(xdim,'relu',h, n_class)

    model_init = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    fname='manifold_model/circle_codim-'+str(codim)+'nclass-'+str(n_class)+'.pth'
    Loss,acc=0,0
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(train_loader):
            
            X, y = X, y
            if xdim==1:
                X=X.unsqueeze(1)
            X=X.float()
            X_adv=X

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

        Loss, acc = 0,0
        train_time = time.time()

        if epoch % 100 == 0:
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n,Loss, acc)
            diff = model.first.weight.data - model_init.first.weight.data
            tmp = torch.matmul(diff,torch.tensor( m_proj,dtype=torch.float ))
            opt.zero_grad()
            model.eval()
            # print(delta)
        
        if epoch >= epochs-1:
            epss = epss = np.array([i*2./10 for i in range(1,100)]+[i**np.sqrt(D/codim) for i in range(1,100)])
            for eps in epss:
                logging.info('%d, %.4f',epoch+1, eps)
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='1')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='2')
                test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='3')
                tmp1 = test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='4')
                tmp = test(model=model, attack=attack, epsilon=eps, LOSS=LOSS,method='5')
                if tmp < 0.01 and tmp1 < 0.01:
                    break

            delta = attack_fgsm(model, X, y, eps_test, False)
            opt.zero_grad()
            delta = delta.cpu()
            delta = torch.matmul(delta, torch.tensor(m_proj,dtype=torch.float))
            logging.info('%.4f',sum(delta[0]**2)/(eps_test**2))
            logging.info('%d',D)

        model.train()
    
    
    return model

def test(model, attack, epsilon, LOSS,print_=False,method='1'):
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    criterion = nn.CrossEntropyLoss()
 
    for i, (X, y) in enumerate(test_loader):
        
        X, y = X, y
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
    
    return acc

#codim_seq=np.arange(0,510,50)
loss_lst=[]
acc_lst=[]
n_class=2

dummy_scalar=1
std_dev=1
epsilon=0.0
noise=1
eps_test=float(sys.argv[4])
D=int(sys.argv[1])
print(D)
k=10
samples=int(int(sys.argv[9])/k)

codim=int(sys.argv[2])
m = np.random.randn(codim,D)/np.sqrt(D)

m_proj = np.linalg.inv(np.matmul(m, m.transpose()) )
m_proj = np.matmul(m.transpose(),m_proj)
m_proj = np.matmul(m_proj, m)

m_res = np.eye(D) - m_proj

# Create gaussian clusters
seed=int(sys.argv[6])
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
beta = np.ones(codim)/np.sqrt(codim)

x=np.random.randn(samples*k,codim)*std_dev/np.sqrt(codim)
y=np.random.randn(samples*k)
mu=[]
shift = []
sign = []
for i in range(k):
    mu.append( np.random.randn(codim) )
    sign.append( i % 2)
    start = int((i-1)*samples)
    end = int(i*samples)
    for j in range(start,end):
        x[j,:] += mu[i] 
        y[j] = sign[i]

x= np.matmul(x, m[:codim,:])

for i in range(k):
    if sys.argv[8] == 'const':
        shift.append(np.random.randn(D)/np.sqrt(D))
    elif sys.argv[8] == 'codim':
        shift.append(np.random.randn(D)/np.sqrt(D)*np.sqrt(codim))
    else:
        shift.append(np.random.randn(D))
    start = int((i-1)*samples)
    end = int(i*samples)
    for j in range(start,end):
        x[j,:] += shift[i]


y = to_onehot(y, 2)

x_test=np.random.randn(1000,codim)*std_dev/np.sqrt(codim)
y_test=np.random.rand(1000) 
for i in range(k):
    start = int((i-1)*(1000/k))
    end = int(i*(1000/k))
    for j in range(start,end):
        x_test[j,:] += mu[i] 
        y_test[j] = sign[i]    

x_test= np.matmul(x_test, m[:codim,:])

for i in range(k):
    start = int((i-1)*(1000/k))
    end = int(i*(1000/k))
    for j in range(start,end):
        x_test[j,:] += shift[i]

y_test = to_onehot(y_test, 3)

print(codim)

train_data=torch.utils.data.TensorDataset(torch.from_numpy(x),y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_data=torch.utils.data.TensorDataset(torch.from_numpy(x_test),y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

lr = float(sys.argv[3])
epochs = int(sys.argv[5])
train(lr,epochs,attack='l2',epsilon=epsilon)
