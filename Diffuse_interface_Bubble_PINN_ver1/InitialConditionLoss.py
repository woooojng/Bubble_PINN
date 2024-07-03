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
    
    u1_initial, u2_initial, P_initial, phi_initial, m_D_initial = net(x, y, t_zero)
    
    #Compute actual initial condition
    #Equation (4.2) in the paper DSB23
    phi_IC = InitialCondition_phi(net, x, y)
    
    u1_IC = InitialCondition_u1(net, x, y)
    u2_IC = InitialCondition_u2(net, x, y)
    
    
    #BinaryCrossEntropyLogit=torch.nn.BCEWithLogitsLoss(weight=None)
    BinaryCrossEntropy=torch.nn.BCELoss(weight=None)
    
    predicted_fluid =  (phi_initial +1)/2
    #predicted_fluid = torch.clamp(predicted_fluid, min = 10**-3, max = 1-10**-3)
    target = (phi_IC +1)/2
    
    phi_IC_loss = BinaryCrossEntropy(predicted_fluid, target)

    
    #u1_error = u1_initial- u1_IC
    #u1_error_normalized = u1_error/(torch.max(torch.abs(u1_error)+.01))
    #u1_IC_loss = mse_cost_function(u1_error_normalized, zero)
    u1_IC_loss = mse_cost_function(u1_initial, u1_IC)
    
    #u2_error = u2_initial- u2_IC
    #u2_error_normalized = u2_error/(torch.max(torch.abs(u2_error))+.01)
    #u2_IC_loss = mse_cost_function(u2_error_normalized, zero)
    u2_IC_loss = mse_cost_function(u2_initial, u2_IC)
    
    #Combine
    initial_domain_loss = phi_IC_loss + u1_IC_loss + u2_IC_loss
       
    return initial_domain_loss

def InitialCondition_phi(net, x, y): #Equation (4.2) in the paper DSB23
    dist = torch.sqrt((x-.5)**2 + (y-.5)**2)
    indicator = dist - .25
    
    #copy the column of 1 x (number of collocation) indicator tensor to make two columns tensor.
    #indicator = torch.cat((indicator,indicator), 1)
    
    #Transform the values into domain1 density 1000 or domain2 density 100 w.r.t. pos/neg distance
    #indicator[:, 0] = torch.where(indicator[:, 0] > 0, net.rho_1, 0)
    #indicator[:, 1] = torch.where(indicator[:, 1] > 0, 0, net.rho_2)
    indicator[:, 0] = torch.where(indicator[:, 0] > 0, np.tanh(-1/(2*net.epsilon_thick)), np.tanh(1/(2*net.epsilon_thick)))

    
    return indicator

def InitialCondition_u1(net, x, y):
    zero = torch.zeros_like(x)
    return zero

def InitialCondition_u2(net, x, y):
    zero = torch.zeros_like(x)
    return zero
    





