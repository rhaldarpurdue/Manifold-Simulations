#from data_utils import *
import torch
import numpy as np
import random
from utils import *
import argparse
import logging
import time
import math
import os
from torchvision.models import resnet18
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="total epochs")
parser.add_argument("--seed", type=int, default=1, help="seed for run")
parser.add_argument("--bs",type=int, default=128, help="batch size")
parser.add_argument("--lr",type=float, default=1e-3, help="learning rate")
parser.add_argument("--res",type=int, default=128, help="resolution (px)")
arg = parser.parse_args()
print(arg)

seed=arg.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)




logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='imagenet_'+str(arg.res)+'_'+str(seed)+'.txt',
                filemode='w',
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)

def robust_train(lr_max,epochs,loader,attack='none',epsilon=0.3,alpha=2/255,trim=True,CL=False):
    attack_iters=int((epsilon/alpha)*1.2)
    model=resnet18().cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        #linf = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=attack_iters, random_start=True)
        for i, (X, y) in enumerate(loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            #X=T.Pad(padding=pad,fill=fill)(X) # adding borders
            if attack == 'fgsm':
                delta=attack_fgsm(model, X, y, epsilon,trim)
            elif attack == 'none' :#or epoch<3:
                delta = torch.zeros_like(X)
            elif attack == 'pgd':#and epoch>=3:
                delta=attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters,1,trim)
                #delta=attack_pgd(model, X, y, epsilon, alpha, attack_iters,1,True)
                #xadv=linf(X,y)
                #delta=xadv-X
            if trim:
                output = model(torch.clamp(X + delta, 0, 1))
            else:
                output = model(X + delta)
            
            if CL:
                loss = criterion(output, y) + model.contrast_loss(X, y)
            else:
                loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        #if train_acc/train_n>0.90: #and epoch>3: #early stopping for adv loss fmnist 0.5000 epochs >5 total
         #   break

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',epoch, train_time - start_time, lr_max, train_loss/train_n, train_acc/train_n)
    return model

def classwise_acc(model,loader,epsilon=0.3,step_size=2/255,attack='linf',trim=True,label=False,return_data=False,ncls=10):
    model.eval()
    attack_iters=int((epsilon/step_size)*1.2)
    acc=0
    n=0
    #rob_lst=[]
    class_acc=[0]*ncls
    class_n=[0]*ncls
    advloss=0
    if return_data:
        x_adv,y_adv=[],[]
    for i, (X, y) in enumerate(loader):
        X, y = X.float().cuda(), y.cuda()
        #X=T.Pad(padding=pad,fill=fill)(X) # adding borders
        robust=torch.zeros(X.shape[0])
        if attack=='linf':
            #delta=attack_fgsm(model, X, y, epsilon,True)
            if label:
                delta=attack_pgd_linf(model, X, y, epsilon, step_size, attack_iters,1,trim,label=True)
            else:
                delta=attack_pgd_linf(model, X, y, epsilon, step_size, attack_iters,1,trim,label=False)
        elif attack=='l2':
            delta=attack_pgd_l2(model, X, y, epsilon, step_size, attack_iters,1,trim)
        elif attack=='none':
            delta=delta = torch.zeros_like(X)
        if not label:
            output = model(torch.clamp(X + delta, 0, 1))
        else:
            output = model(X + delta,y)
        if return_data:
            x_adv.append(X+delta)
            y_adv.append(y)
        advloss+=F.cross_entropy(output,y,reduction='sum').item()
        I = output.max(1)[1] == y # index which were not fooled
        robust[I]=1
        for j in range(ncls):
            cls_id=(y==j).cpu()
            class_acc[j]+=robust[cls_id].sum().item()
            class_n[j]+=cls_id.sum().item()
        acc += robust.sum().item()
        n += y.size(0)
        #rob_lst.append(robust)
    logger.info('Adv Loss \t Adv Acc')
    logger.info('%.4f \t %.4f', advloss/n, (acc)/n)
    class_acc = [class_acc[index]/value for index,value in enumerate(class_n)]
    if return_data:
        x_adv=torch.cat(x_adv)
        y_adv=torch.cat(y_adv)
        return class_acc, torch.utils.data.TensorDataset(x_adv,y_adv)
    else:
        return class_acc,acc/n, advloss/n


result=[]
resize = transforms.Resize(size=(arg.res,arg.res))
Transforms = transforms.Compose([resize,transforms.ToTensor()])
data_train=torchvision.datasets.ImageFolder('data/train',transform=Transforms)
data_test=torchvision.datasets.ImageFolder('data/test',transform=Transforms)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=arg.bs, shuffle=True) 
test_loader = torch.utils.data.DataLoader(data_test, batch_size=arg.bs, shuffle=False)

mod=robust_train(lr_max=arg.lr,epochs=arg.epochs,loader=train_loader,attack="none",epsilon=0.3,trim=False)
for eps in [16*i/(255*30) for i in range(30)]:
    _,r,_=classwise_acc(mod,test_loader,trim=True,attack='linf',epsilon=eps,ncls=2)
    result.append(r)

import pickle as pkl
with open('imagenet'+str(arg.res)+'_'+str(arg.seed)+'.pkl', "wb") as f:
    pkl.dump(result, f)


