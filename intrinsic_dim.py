import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F_vis
import skdim


mnist_train = datasets.MNIST("../../mnist-data", train=True, download=True, transform=transforms.ToTensor())
cifar_train= datasets.CIFAR10("../../cifar10-data", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=50000, shuffle=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=60000, shuffle=True)
label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for x_temp,y_temp in train_loader:
    break
x_temp=x_temp.flatten(start_dim=1, end_dim=- 1)
ids=y_temp==9
x_sub=x_temp[ids]
x_temp=np.array(x_temp)

#estimate global intrinsic dimension
#danco = skdim.id.DANCo().fit(x_temp)
#estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
lpca = skdim.id.lPCA().fit_pw(x_temp,
                              n_neighbors = 100,
                              n_jobs = 1)

#get estimated intrinsic dimension
print(danco.dimension_, np.mean(lpca.dimension_pw_))

pca=skdim.id.lPCA()
mle=skdim.id.MLE()
twonn=skdim.id.TwoNN()
#danco=skdim.id.DANCo()
mom=skdim.id.MOM()
#mada=skdim.id.MADA()

d=pca.fit(x_temp).dimension_
d=mle.fit(x_temp).dimension_
twonn.fit(x_temp).dimension_
mom.fit(x_temp).dimension_
#d=38,9 mnist cifar lpca
#d=13.36, 27.656 mnist cifar mle k=5
#d=14.90 , 31.65 mnist cifar two nn
#d=13.84,  mnist cifar mom

# cifar 10 label=0 lpca,mle,two nn (8, 18.34, 21.77)
# cifar 10 label=1 lpca,mle,two nn (11, 21.20, 25.98)
# cifar 10 label=2 lpca,mle,two nn (8, 21.35, 25.53)
# cifar 10 label=3 lpca,mle,two nn (11, 21.08, 26.09)
# cifar 10 label=4 lpca,mle,two nn (9, 21.43, 24.60)
# cifar 10 label=5 lpca,mle,two nn (13, 22.11, 27.32)
# cifar 10 label=6 lpca,mle,two nn (7, 22.96, 26.19)
# cifar 10 label=7 lpca,mle,two nn (14, 22.57, 25.95)
# cifar 10 label=8 lpca,mle,two nn (10, 19.80, 24.77)
# cifar 10 label=9 lpca,mle,two nn (16, 24.42, 29.11)

# MNIST label=0 lpca,mle,two nn (19,13.01,15.33)
# MNIST label=1 lpca,mle,two nn (8, 9.65, 12.98)
# MNIST label=2 lpca,mle,two nn (31, 14.14, 15.29)
# MNIST label=3 lpca,mle,two nn (29, 15.14, 16.35)
# MNIST label=4 lpca,mle,two nn (27, 13.65, 14.71)
# MNIST label=5 lpca,mle,two nn (23, 14.70, 15.90)
# MNIST label=6 lpca,mle,two nn (21, 12.65, 14.02)
# MNIST label=7 lpca,mle,two nn (21, 12.08, 13.39)
# MNIST label=8 lpca,mle,two nn (32, 15.65, 16.60)
# MNIST label=9 lpca,mle,two nn (22, 12.79, 14.03)

twonn.fit(x_sub).dimension_
mle.fit(x_sub).dimension_
pca.fit(x_sub).dimension_
