#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#from torch.autograd import Variable

def y_center_of_mass(net, t):
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u
    
    Domain_collocation = 20000
    
    #Pick IC/NSpde Condition Training Random Points in Numpy
    x = np.random.uniform(low= x_l, high=x_u, size=(Domain_collocation, 1)) 
    y = np.random.uniform(low= y_l, high=y_u, size=(Domain_collocation, 1)) 
    
    #Move to pytorch tensors
    x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
    
    
    #Pick IC Training t starting points to make tensor
    t = Variable(t * torch.ones_like(x), requires_grad=True).to(device)
   
    u1, u2, P, rho = net(x, y, t)
   
    #Eliminate values at domain with rho >(rho1+rho2)/2
    #xy_domain_pt = torch.cat(torch.zeros_like(x), y, 1)
    #double_rho_on_domain =  torch.cat((0, rho), 1)
    y_unit = torch.where(rho ==1000, 0, 1)
    y = torch.where(rho ==1000, 0, y)

    #Compute the denomenator =int_{Omega2} 1 dx
    omega2_area = 1/Domain_collocation *(torch.sum(y_unit))
    
    
    #Compute the nominator =int_{Omega2} x dx
    omega2_exp = 1/Domain_collocation *torch.sum(y)
    
    
    
    return omega2_exp/omega2_area
