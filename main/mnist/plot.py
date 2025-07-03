# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:03:21 2023

@author: rajde
"""

import pickle 
import numpy as np
import matplotlib.pyplot as plt

pad=0
seed=1
# Open the file in binary mode 
fn='mnist'+str(pad)+'_'+str(seed)+'.pkl'
#fn='imagenet'+str(pad)+'_'+str(seed)+'.pkl'
with open(fn, 'rb') as file: 
      
    # Call load method to deserialze 
    myvar = pickle.load(file) 
  
    print(myvar) 

def get_data(pad): # for mnist or fmnist plot
    
    for seed in range(1,31):
        fn='mnist'+str(pad)+'_'+str(seed)+'.pkl'
        #fn='imagenet'+str(pad)+'_'+str(seed)+'.pkl'
        with open(fn, 'rb') as file: 
              
            # Call load method to deserialze 
            myvar = pickle.load(file) 
        if seed==1:
            eps=np.array(myvar)[:,0]
            x=np.array(myvar)[:,1]
            x=x.reshape(30,1)
        else:
            tmp=np.array(myvar)[:,1]
            tmp=tmp.reshape(30,1)
            x=np.hstack((x,tmp))
    return eps, np.mean(x,axis=1), np.std(x,axis=1)

def get_data2(pad): # use get data2 for imagenet  plot
    
    for seed in range(1,31):
        #fn='fmnist'+str(pad)+'_'+str(seed)+'.pkl'
        fn='imagenet'+str(pad)+'_'+str(seed)+'.pkl'
        with open(fn, 'rb') as file: 
              
            # Call load method to deserialze 
            myvar = pickle.load(file) 
        if seed==1:
            eps=np.array([16*i/(255*30) for i in range(30)])
            #eps=np.array(myvar)[:,0]
            x=np.array(myvar)
            x=x.reshape(30,1)
        else:
            tmp=np.array(myvar)
            tmp=tmp.reshape(30,1)
            x=np.hstack((x,tmp))
    return eps, np.mean(x,axis=1), np.std(x,axis=1)

fig, ax = plt.subplots(figsize=(5, 4))

# standard error bars
d=[0,5,10,20,50] #d is essentially the paddings for fminst or mnist or resolution in case of imagenet plot.
#d=[32,64,128,256,320]
#d=[64,128,320]
for pad in d:
    eps, mean, std=get_data(pad)
    ax.errorbar(eps[3:], mean[3:], yerr=std[3:],marker='x',ls='-', capsize=5, capthick=1)
plt.title('Robustness Mnist when training for 10 epcohs, Linf')
plt.xlabel('eps')
plt.ylim(-0.05,0.72)
#plt.xlim(0.01,)
plt.ylabel('acc')
plt.legend(d,title='Padding')
#plt.legend(d,title='Resolution')
plt.grid()
plt.savefig('mnist_linf', dpi=150)
#plt.show()