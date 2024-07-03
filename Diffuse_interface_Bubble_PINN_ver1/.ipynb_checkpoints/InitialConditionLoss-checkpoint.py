#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lossIC(net, x, y, t_zero):
    mse_cost_function = torch.nn.MSELoss(reduction='mean')
    
    #t_zero = Variable(torch.zeros_like(x), requires_grad=True).to(device)
    #zero_x = torch.zeros_like(x).to(device)
    #Compute estimated initial condition
    u1_initial, u2_initial, P_initial, rho_initial = net(x, y, t_zero)
    
    #Compute actual initial condition
    rho_IC = InitialCondition_rho(net, x, y)
    
    u1_IC = InitialCondition_u1(net, x, y)
    u2_IC = InitialCondition_u2(net, x, y)
    
    #Compute mse
    #################################################################
    #rho_error = rho_initial- rho_IC
    #################################################################
    #rho_error_normalized = rho_error/(torch.max(torch.abs(rho_error))+.01)
    #rho_IC_loss = mse_cost_function(rho_error_normalized, zero)
    #zero_rho = torch.zeros_like(rho_initial).to(device)
    
    #cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 10]).to(device))
    #rho_IC_loss = cross_entropy_loss(rho_initial, rho_IC)

    #BinaryCrossEntropyLogit=torch.nn.BCEWithLogitsLoss(weight=None)
    BinaryCrossEntropy=torch.nn.BCELoss(weight=None)
    predicted_fluid = (1000 - rho_initial)/900
    
    target = (1000 - rho_IC)/900
    
    rho_IC_loss = BinaryCrossEntropy(predicted_fluid, target)

    
    #u1_error = u1_initial- u1_IC
    #u1_error_normalized = u1_error/(torch.max(torch.abs(u1_error)+.01))
    #u1_IC_loss = mse_cost_function(u1_error_normalized, zero)
    u1_IC_loss = mse_cost_function(u1_initial, u1_IC)
    
    #u2_error = u2_initial- u2_IC
    #u2_error_normalized = u2_error/(torch.max(torch.abs(u2_error))+.01)
    #u2_IC_loss = mse_cost_function(u2_error_normalized, zero)
    u2_IC_loss = mse_cost_function(u2_initial, u2_IC)
    
    #Combine
    initial_domain_loss = rho_IC_loss + u1_IC_loss + u2_IC_loss
       
    return initial_domain_loss

def InitialCondition_rho(net, x, y):
    dist = torch.sqrt((x-.5)**2 + (y-.5)**2)
    indicator = dist - .25
    
    #copy the column of 1 x (number of collocation) indicator tensor to make two columns tensor.
    #indicator = torch.cat((indicator,indicator), 1)
    
    #Transform the values into domain1 density 1000 or domain2 density 100 w.r.t. pos/neg distance
    #indicator[:, 0] = torch.where(indicator[:, 0] > 0, net.rho_1, 0)
    #indicator[:, 1] = torch.where(indicator[:, 1] > 0, 0, net.rho_2)
    indicator[:, 0] = torch.where(indicator[:, 0] > 0, net.rho_1, net.rho_2)

    
    return indicator

def InitialCondition_u1(net, x, y):
    zero = torch.zeros_like(x)
    return zero

def InitialCondition_u2(net, x, y):
    zero = torch.zeros_like(x)
    return zero
    

# In[ ]:




