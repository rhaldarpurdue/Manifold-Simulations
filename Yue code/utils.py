import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset

# from dataset import load_dataset

from net import *
import matplotlib.pyplot as plt
import os.path
import argparse

#trim=False
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon,trim):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    if trim:
        delta=clamp(delta,0-X,1-X)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,trim):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            #delta.data = d # used for loss computation
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd_linf(model, X, y, epsilon, alpha, attack_iters, restarts,trim):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data = d # used for loss computation
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts,trim):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    batch_size=X.shape[0]
    channels=len(X.shape)-1
    eps_for_division=1e-10
    shape=(batch_size,)+(1,)*channels
    shape2=(-1,)+(1,)*channels
    for _ in range(restarts):
        # Starting at a uniformly random point
        delta = torch.zeros_like(X).normal_().cuda()
        d_flat = delta.view(X.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(shape)
        # r = torch.zeros_like(n).uniform_(0, 1)
        # delta *= r/n*epsilon
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + eps_for_division
            grad = grad / grad_norms.view(shape)
            d = delta + alpha *grad
            d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1)
            factor = epsilon / d_norms
            factor = torch.min(factor, torch.ones_like(d_norms))
            d = d * factor.view(shape2)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data = d # used for loss computation
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd_linf_mse(model, X, y, epsilon, alpha, attack_iters, restarts,trim):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if output.shape[1]==1:
                loss = F.mse_loss(output.squeeze(), y)
            else:
                loss = F.mse_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data = d # used for loss computation
            delta.grad.zero_()
        if output.shape[1]==1:
            all_loss = F.mse_loss(model(X+delta).squeeze(), y, reduction='none')
        else:
            all_loss = F.mse_loss(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd_l2_mse(model, X, y, epsilon, alpha, attack_iters, restarts,trim):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    batch_size=X.shape[0]
    channels=len(X.shape)-1
    eps_for_division=1e-10
    shape=(batch_size,)+(1,)*channels
    shape2=(-1,)+(1,)*channels
    # print("Calculate attack")
    for _ in range(restarts):
        # Starting at a uniformly random point

        # delta = torch.zeros_like(X).normal_().cuda()
        
        
        delta = torch.zeros_like(X).cuda()
        d_flat = delta.view(X.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(shape)
        # r = torch.zeros_like(n).uniform_(0, 1)
        # delta *= r/n*epsilon
        if trim:
            delta.data = clamp(delta, 0-X, 1-X)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if output.shape[1]==1:
                loss = F.mse_loss(output.squeeze(), y)
            else:
                loss = F.mse_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + eps_for_division
            grad = grad / grad_norms.view(shape)
            d = delta + alpha *grad
            d_norms = torch.norm(d.view(batch_size, -1), p=2, dim=1)
            factor = epsilon / d_norms
            factor = torch.min(factor, torch.ones_like(d_norms))
            d = d * factor.view(shape2)
            if trim:
                d = clamp(d, 0-X, 1-X)
            delta.data = d # used for loss computation
            delta.grad.zero_()
        if output.shape[1]==1:
            all_loss = F.mse_loss(model(X+delta).squeeze(), y, reduction='none')
        else:
            all_loss = F.mse_loss(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta