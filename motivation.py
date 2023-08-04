# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:37:41 2023

@author: rajde
"""

x=mnist_train[0][0]
y=torch.tensor(mnist_train[0][1])
show(make_grid(x))

x2=mnist_train[9][0]
show(make_grid(x2))
delta=x2-x
delta=torch.clamp(delta,0,0.5)

x_adv_nat=torch.clamp(x+delta,0,1)
show(make_grid(x_adv_nat))
mod(x_adv_nat.cuda().unsqueeze(0)).max(1)[1]

d2=attack_pgd_linf(mod, x.cuda().unsqueeze(0), y.unsqueeze(0).cuda(), 0.1, 0.01, 50,10,True)
x_adv=torch.clamp(x+d2.detach().cpu(),0,1)
show(make_grid(x_adv))
mod(x_adv.cuda()).max(1)[1]
  