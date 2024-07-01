#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
#import numpy as np
#from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#from torch.autograd import Variable

def lossNSpde(net, x, y, t):
    mse_cost_function = torch.nn.MSELoss(reduction='mean')
    # The dependent variable 'u1, u2, P, p, m' are obtained from the network based on independent variables x and t.
    u1, u2, P, p = net(x, y, t)
    #p = p[:,:1] + p[:,-1:]
    m = net.mu(x,y,t)
    
    # Compute \nabla u
    # Before defining the PDE = d(u1)/dt - d^2(u1)/(dx)^2, we need to compute d(u1)/dx and d(u1)/dt.
    # torch.autograd.grad <- Computes and returns the sum of gradients of outputs with respect to the inputs.
    
    #Compute Derivatives
    u1_x = torch.autograd.grad(u1.sum(), x,create_graph=True)[0] #Compute u1_x
    u1_y = torch.autograd.grad(u1.sum(), y,create_graph=True)[0] #Compute u1_y
    u1_t = torch.autograd.grad(u1.sum(), t,create_graph=True)[0] #Compute u1_t
    
    u2_x = torch.autograd.grad(u2.sum(), x,create_graph=True)[0] #Compute u2_x
    u2_y = torch.autograd.grad(u2.sum(), y,create_graph=True)[0] #Compute u2_y
    u2_t = torch.autograd.grad(u2.sum(), t,create_graph=True)[0] #Compute u2_t
    
    m_x = torch.autograd.grad(m.sum(), x,create_graph=True)[0] #Compute m_x
    m_y = torch.autograd.grad(m.sum(), y,create_graph=True)[0] #Compute m_y
    
    p_x = torch.autograd.grad(p.sum(), x,create_graph=True)[0] #Compute p_x
    p_y = torch.autograd.grad(p.sum(), y,create_graph=True)[0] #Compute p_y
    
    P_x = torch.autograd.grad(P.sum(), x,create_graph=True)[0] # Compute P_x
    P_y = torch.autograd.grad(P.sum(), y,create_graph=True)[0] #P_y
    
    #second derivatives
    u1_xx = torch.autograd.grad(u1_x.sum(),x,create_graph=True)[0] #Compute u1_xx
    u1_xy = torch.autograd.grad(u1_x.sum(),y,create_graph=True)[0] #Compute u1_xy
    u1_yy = torch.autograd.grad(u1_y.sum(),y,create_graph=True)[0] #Compute u1_yy
    
    u2_xx = torch.autograd.grad(u2_x.sum(),x,create_graph=True)[0] #Compute u2_xx
    u2_xy = torch.autograd.grad(u2_x.sum(),y,create_graph=True)[0] #Compute u2_xy
    u2_yy = torch.autograd.grad(u2_y.sum(),y,create_graph=True)[0] #Compute u2_yy

    
    #Define normal vector n = \nabla p / |\nabla p|
    gradp = torch.cat((p_x, p_y), 1) #.clamp(min=10**-5,max=10**5)
    
    gradp_norm = torch.cdist(gradp, torch.zeros(1, 2).to(device), p=1)
    
    
    #Compute Tangent Hyperbolic function
    #to ignore only moving boundary points through the domain.
    
    MvBdryCoefficient = 1/(.1 * (gradp_norm)+ torch.ones_like(gradp_norm))
    
    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    g = 0.98
    NSpde_xcoord = (p*( u1_t + u1 * u1_x + u2 * u1_y) + P_x - (2 * m_x * u1_x + m_y * u2_x + m_y * u1_y)- m * (2*u1_xx +u2_xy + u1_yy) - p_x * g)
    NSpde_ycoord = (p*( u2_t + u1 * u2_x + u2 * u2_y) + P_y - (m_x * u1_y + 2 * m_y * u2_y + m_x * u2_x)- m * (u1_xy +2*u2_yy + u2_xx) - p_y * g)
    
    NSpde_div_xcoord = u1_x + u1_y
    NSpde_div_ycoord = u2_x + u2_y
    
    zero_NSpde_xcoord = torch.zeros_like(NSpde_xcoord)
    zero_NSpde_ycoord = torch.zeros_like(NSpde_ycoord)
    
    zero_NSpde_div_xcoord = torch.zeros_like(NSpde_div_xcoord)
    zero_NSpde_div_ycoord = torch.zeros_like(NSpde_div_ycoord)
    
    # Define MSE Function on inner space
    NSpde_loss = (mse_cost_function(MvBdryCoefficient*NSpde_xcoord, zero_NSpde_xcoord) + mse_cost_function(MvBdryCoefficient*NSpde_ycoord, zero_NSpde_ycoord))
    NSpde_div_loss = (mse_cost_function(MvBdryCoefficient*NSpde_div_xcoord, zero_NSpde_div_xcoord) + mse_cost_function(MvBdryCoefficient*NSpde_div_ycoord, zero_NSpde_div_ycoord))
    
    # Normalize
    #NSpde_loss = (NSpde_loss - NSpde_loss.mean())/(1+NSpde_loss.std())
    #NSpde_div_loss = (NSpde_div_loss - NSpde_div_loss.mean())/(1+NSpde_div_loss.std())

    # Combine
    NS_loss = (NSpde_loss + NSpde_div_loss)
    
    return NS_loss

