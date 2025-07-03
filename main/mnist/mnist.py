import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import datasets, transforms
# import torchvision.transforms as T

# from utils import *
import argparse
import logging
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument("--pad", type=int, default=0, help="padding number")
parser.add_argument("--seed", type=int, default=1, help="seed for run")
parser.add_argument("--attack", type=str, default='linf',help="l2 or linf attack")
arg = parser.parse_args()
print(arg)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# CNN classifier
def CNN(pad=0):
    print(pad)
    c=math.floor(pad[0]/2)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*(7+c)*(7+c),100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model
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
data_loc="~/mnist"
mnist_train = datasets.MNIST(data_loc, train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(data_loc, train=False, download=True, transform=transforms.ToTensor())


def attack_fgsm(model, X, y, epsilon,trim):
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

        d = delta + alpha *grad/(torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(shape2)+0.0000001)
        d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1).detach()
        factor = epsilon / (d_norms+0.0000001)
        factor = torch.min(factor, torch.ones_like(d_norms))
        d = d * factor.view(shape2)
        
        delta.data = d # used for loss computation
        delta.grad.zero_()
        # manually change this part for on-manifold attack:
    model.train()
    return delta.detach()


# Test and train loader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True) #adjust batch size based on using gradient accumulation or not
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False)

# Train model 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

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
            X=F.pad(X,pad,value=fill) # adding borders
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
    return model



# Adversarial Accuracy
def pgd_attacks(model,pad=0,fill=0.5,epsilon=0.3,step_size=0.01,attack_iters=50,attack='l2'):
    acc=0
    n=0
    rob_lst=[]
    advloss=0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X=F.pad(X,pad,value=fill) # adding borders
        robust=torch.zeros(X.shape[0])
        if attack=='linf':
            delta=attack_pgd_linf(model, X, y, epsilon, step_size, attack_iters,10,True)
        elif attack=='l2':
            delta=attack_fgsm(model, X, y, epsilon, True)
        output=model(torch.clamp(X+delta,0,1))
        advloss+=F.cross_entropy(output,y,reduction='sum')
        I = output.max(1)[1] == y # index which were not fooled
        robust[I]=1
        acc += robust.sum().item()
        n += y.size(0)
        rob_lst.append(robust)
    logger.info('%.4f \t %.4f \t %.4f',epsilon, advloss/n, (acc)/n)
    return rob_lst ,(epsilon,acc/n)

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


# setting border parameters
a = arg.pad
pad=(a,a,a,a) # how many extra pixels we are adding at the border in each direction; For example 28X28X1->(28+2*pad)X(28+2*pad)X1
fill=0.5 # fill is btw 0-1; corresponding to the pixel values at the border 1=white 0=black
print('D:',a,';Seed:',seed)

# Training and adv robustness
mod=robust_train(1e-3, 10,pad=pad,fill=fill,epsilon=0)
res=[]
logger.info('Eps \t Adv Loss \t Adv Acc')
for eps in [i*0.1 for i in range (41) ]: # change the eps range based on your attack for linf 0->0.3 eps is suitable for mnist
    #for l2 mnist suitable range of eps is 0->4.0
    _,r=pgd_attacks(mod,pad=pad,fill=fill,epsilon=eps,attack=arg.attack)
    res.append(r)

import pickle as pkl
with open('mnist'+str(arg.pad)+'_'+str(arg.seed)+'.pkl', "wb") as f:
    pkl.dump(res, f)