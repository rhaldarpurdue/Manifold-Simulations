import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import datasets, transforms
# import torchvision.transforms as T
import numpy as np
import random
# from utils import *
import argparse
import logging
import time
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pad", type=int, default=0, help="padding number")
parser.add_argument("--seed", type=int, default=1, help="seed for run")
parser.add_argument("--attack", type=str, default='linf',help="l2 or linf attack")
arg = parser.parse_args()
print(arg)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN(nn.Module):
    def __init__(self,pad=0):
        super().__init__()
        self.c=math.floor(pad[0]/2)
        self.l1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.l2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.l3 = nn.Linear(32*(7+self.c)*(7+self.c),100)
        self.l4 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.flatter = Flatten()
    def forward(self, x):
        # print(x.shape)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.flatter(x)
        # print(x.shape)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        return x


# Fully connected classifier
def FCN(pad=0):
    dim=pad+28
    model = nn.Sequential(Flatten(),
        nn.Linear(dim*dim,100),
        nn.ReLU(),
        nn.Linear(100,75),
        nn.ReLU(),
        nn.Linear(75, 10)
    )
    return model

# Read data
data_loc="~/fmnist"
fmnist_train = datasets.FashionMNIST(data_loc, train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(data_loc, train=False, download=True, transform=transforms.ToTensor())


def attack_fgsm(model, X, y, epsilon,trim):
    model.eval()
    attack_iters = 20
    # attack_iters = 50
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

        d = delta + alpha *grad/(torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(shape2)+0.0000001)
        d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1).detach()
        factor = epsilon / (d_norms+0.0000001)
        factor = torch.min(factor, torch.ones_like(d_norms))
        d = d * factor.view(shape2)
        
        delta.data = torch.clamp(X+d, 0, 1) - X # used for loss computation
        delta.grad.zero_()
        # delta = torch.clamp(X+d, 0, 1) - X
        # manually change this part for on-manifold attack:
    model.train()
    return delta.detach()


def attack_fgsm_linf(model, X, y, epsilon,trim):
    model.eval()
    # attack_iters = 20
    attack_iters = 50
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

        d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        
        delta.data = d # used for loss computation
        delta.grad.zero_()
        # delta = torch.clamp(X+d, 0, 1) - X
        # manually change this part for on-manifold attack:
    model.train()
    return delta.detach()

# Test and train loader
train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=256, shuffle=True) #adjust batch size based on using gradient accumulation or not
test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=256, shuffle=False)

#arg=['0','1'] # D, seed
# Train model 
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='fmnist_'+str(arg.pad)+'_'+str(arg.seed)+'_linf.txt',
                filemode='w',
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)

def robust_train(lr_max,epochs,pad=0,fill=0.5,attack='none',epsilon=0.3,attack_iters=50,alpha=0.01):
    model=CNN(pad=pad).cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
            
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X=F.pad(X,pad,value=fill,mode="constant") # adding borders
            if epsilon >0:
                if attack == 'fgsm':
                    delta=attack_fgsm(model, X, y, epsilon,True)
                elif attack == 'none':
                    delta = torch.zeros_like(X)
                elif attack == 'pgd':
                    delta=attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters,10,True)    
                output = model(torch.clamp(X + delta, 0, 1))
            else:
                output = model(torch.clamp(X, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',epoch, train_time - start_time, lr_max, train_loss/train_n, train_acc/train_n)
        if train_loss/train_n < 0.02:
            break
    return model



# Adversarial Accuracy
def pgd_attacks(model,pad=0,fill=0.5,epsilon=0.3,step_size=0.01,attack_iters=50,attack='l2'):
    acc=0
    n=0
    rob_lst=[]
    advloss=0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X=F.pad(X,pad,value=fill,mode="constant") # adding borders
        robust=torch.zeros(X.shape[0])
        if attack=='linf':
            delta=attack_fgsm_linf(model, X, y, epsilon, True)
        elif attack=='l2':
            delta=attack_fgsm(model, X, y, epsilon, True)
        output=model(torch.clamp(X+delta,0,1))
        advloss+=F.cross_entropy(output,y,reduction='sum').item()
        I = output.max(1)[1] == y # index which were not fooled
        robust[I]=1
        acc += robust.sum().item()
        n += y.size(0)
        rob_lst.append(robust)
    logger.info('%.4f \t %.4f \t %.4f',epsilon, advloss/n, (acc)/n)
    # print(epsilon, ((torch.clamp(X+delta,0,1)-X)[1,:,:,:]**2).sum() )
    return rob_lst,(epsilon,acc/n)

# Usage: show(make_grid(X)) ; will plot the digits
from torchvision.utils import make_grid
import torchvision.transforms.functional as F_vis
import matplotlib.pyplot as plt

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F_vis.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


seed=arg.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# setting border parameters
a = arg.pad
pad=(a,a,a,a) # how many extra pixels we are adding at the border in each direction; For example 28X28X1->(28+2*pad)X(28+2*pad)X1
fill=0.5 # fill is btw 0-1; corresponding to the pixel values at the border 1=white 0=black
print('D:',a,';Seed:',seed)

# Training and adv robustness
mod=robust_train(1e-3, 30,pad=pad,fill=fill,epsilon=0)
res=[]
logger.info('Eps \t Adv Loss \t Adv Acc')
# for eps in [0.0, 0.5,1.0, 1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]:
for eps in [i/300 for i in range(30)]:
# for eps in [1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]:
    _,r=pgd_attacks(mod,pad=pad,fill=fill,epsilon=eps,attack=arg.attack)
    res.append(r)

import pickle as pkl
with open('fmnist'+str(arg.pad)+'_'+str(arg.seed)+'.pkl', "wb") as f:
    pkl.dump(res, f)

