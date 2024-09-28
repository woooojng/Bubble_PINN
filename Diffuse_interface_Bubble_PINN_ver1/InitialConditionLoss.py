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
    
    u1_initial, u2_initial, P_initial, phi_initial, mu_initial = net(x, y, t_zero)
    
    #Compute actual initial condition
    #Equation (4.2) in the paper DSB23
    phi_IC = InitialCondition_phi(net, x, y)
    
    u1_IC = InitialCondition_u1(net, x, y)
    u2_IC = InitialCondition_u2(net, x, y)
    '''
    BinaryCrossEntropy=torch.nn.BCELoss(weight=None)
    
    predicted_fluid =  (phi_initial +1)/2

    target = (phi_IC +1)/2
    '''
    phi_IC_loss = mse_cost_function(phi_initial, phi_IC)

    u1_IC_loss = mse_cost_function(u1_initial, u1_IC)
    
    u2_IC_loss = mse_cost_function(u2_initial, u2_IC)
    
    #Combine
    initial_domain_loss = phi_IC_loss + u1_IC_loss + u2_IC_loss
       
    return initial_domain_loss

def InitialCondition_phi(net, x, y): #Equation (4.2) in the paper DSB23
    dist_from_center = torch.sqrt((x-.5)**2 + (y-.5)**2)
    signed_dist = .25 - dist_from_center
    
    #copy the column of 1 x (number of collocation) indicator tensor to make two columns tensor.
    #indicator = torch.cat((indicator,indicator), 1)
    
    #Transform the values into domain1 density 1000 or domain2 density 100 w.r.t. pos/neg distance
    dist = torch.tanh(signed_dist/(np.sqrt(2)*net.epsilon))
    #torch.where(dist_from_bdry < 0, np.tanh(-1/(np.sqrt(2)*net.epsilon)), np.tanh(1/(np.sqrt(2)*net.epsilon)))
    # tanh(.25/(sqrt(2)*.04) ) =0.99971005884 tanh( 1/(sqrt(2)*.1) ) = 0.99999855729...

    
    return dist

def InitialCondition_u1(net, x, y):
    zero = torch.zeros_like(x)
    return zero

def InitialCondition_u2(net, x, y):
    zero = torch.zeros_like(x)
    return zero
    





