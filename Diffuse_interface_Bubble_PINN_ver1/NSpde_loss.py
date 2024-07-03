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
    u1, u2, P, phi, m_D = net(x, y, t)
    
    m = net.mu(x,y,t)
    rho = net.rho(x,y,t)
    J1, J2 = net.J(x,y,t)
    tau11, tau12, tau21, tau22 = net.tau(x,y,t)
    zeta11, zeta12, zeta21, zeta22 = net.zeta(x,y,t)

    mobility = net.mobility
    epsilon_thick = net.epsilon_thick
    sigma = net.sigma

    # Compute \nabla u
    
    #Compute Derivatives
    u1_x = torch.autograd.grad(u1.sum(), x,create_graph=True)[0] #Compute u1_x
    u1_y = torch.autograd.grad(u1.sum(), y,create_graph=True)[0] #Compute u1_y
    u1_t = torch.autograd.grad(u1.sum(), t,create_graph=True)[0] #Compute u1_t
    
    u2_x = torch.autograd.grad(u2.sum(), x,create_graph=True)[0] #Compute u2_x
    u2_y = torch.autograd.grad(u2.sum(), y,create_graph=True)[0] #Compute u2_y
    u2_t = torch.autograd.grad(u2.sum(), t,create_graph=True)[0] #Compute u2_t
    
    m_x = torch.autograd.grad(m.sum(), x,create_graph=True)[0] #Compute m_x
    m_y = torch.autograd.grad(m.sum(), y,create_graph=True)[0] #Compute m_y
    
    phi_x = torch.autograd.grad(phi.sum(), x,create_graph=True)[0] #Compute phi_x
    phi_y = torch.autograd.grad(phi.sum(), y,create_graph=True)[0] #Compute phi_y
    phi_t = torch.autograd.grad(phi.sum(), t,create_graph=True)[0]  
    
    P_x = torch.autograd.grad(P.sum(), x,create_graph=True)[0] # Compute P_x
    P_y = torch.autograd.grad(P.sum(), y,create_graph=True)[0] # Compute P_y

    m_D_x = torch.autograd.grad(m_D.sum(), x,create_graph=True)[0] 
    m_D_y = torch.autograd.grad(m_D.sum(), y,create_graph=True)[0]

    J1_x = torch.autograd.grad(J1.sum(), x,create_graph=True)[0] 
    J1_y = torch.autograd.grad(J1.sum(), y,create_graph=True)[0] 
    J2_x = torch.autograd.grad(J2.sum(), x,create_graph=True)[0] 
    J2_y = torch.autograd.grad(J2.sum(), y,create_graph=True)[0] 

    tau11_x = torch.autograd.grad(tau11.sum(), x,create_graph=True)[0]
    tau12_x = torch.autograd.grad(tau12.sum(), x,create_graph=True)[0] 
    tau21_y = torch.autograd.grad(tau21.sum(), y,create_graph=True)[0]
    tau22_y = torch.autograd.grad(tau22.sum(), y,create_graph=True)[0] 

    zeta11_x = torch.autograd.grad(zeta11.sum(), x,create_graph=True)[0] 
    zeta12_x = torch.autograd.grad(zeta12.sum(), x,create_graph=True)[0]
    zeta21_y = torch.autograd.grad(zeta21.sum(), y,create_graph=True)[0] 
    zeta22_y = torch.autograd.grad(zeta22.sum(), y,create_graph=True)[0] 

    rho_x = torch.autograd.grad(rho.sum(), x,create_graph=True)[0]  
    rho_y = torch.autograd.grad(rho.sum(), y,create_graph=True)[0]
    rho_t = torch.autograd.grad(rho.sum(), t,create_graph=True)[0]  
    
    
    #second derivatives
    u1_xx = torch.autograd.grad(u1_x.sum(),x,create_graph=True)[0] #Compute u1_xx
    u1_xy = torch.autograd.grad(u1_x.sum(),y,create_graph=True)[0] #Compute u1_xy
    u1_yy = torch.autograd.grad(u1_y.sum(),y,create_graph=True)[0] #Compute u1_yy
    
    u2_xx = torch.autograd.grad(u2_x.sum(),x,create_graph=True)[0] #Compute u2_xx
    u2_xy = torch.autograd.grad(u2_x.sum(),y,create_graph=True)[0] #Compute u2_xy
    u2_yy = torch.autograd.grad(u2_y.sum(),y,create_graph=True)[0] #Compute u2_yy

    phi_xx = torch.autograd.grad(phi_x.sum(),x,create_graph=True)[0] 
    phi_yy = torch.autograd.grad(phi_y.sum(),y,create_graph=True)[0]

    m_D_xx = torch.autograd.grad(m_D_x.sum(),x,create_graph=True)[0] 
    m_D_yy = torch.autograd.grad(m_D_y.sum(),y,create_graph=True)[0] 
    
    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    g = 9.8

    #(2.1a)

    RHS_xcoord = rho * g * torch.ones_like(u1)
    RHS_ycoord = rho * g * torch.ones_like(u1)

    LHS1_xcoord = rho_t * u1 + rho * u1_t
    LHS1_ycoord = rho_t * u2 + rho * u2_t

    LHS2_xcoord = rho * (2*u1*u1_x + u1_y*u2 + u1*u2_y)
    LHS2_ycoord = rho * (u1_x*u2 + u1*u2_x + 2*u2*u2_y)

    LHS3_xcoord = u1_x*J1 + u1*J1_x + u2_y*J1 + u2*J1_y 
    LHS3_ycoord = u1_x*J2 + u1*J2_x + u2_y*J2 + u2*J2_y

    LHS4_xcoord = P_x
    LHS4_ycoord = P_y

    LHS5_xcoord = tau11_x + tau21_y
    LHS5_ycoord = tau12_x + tau22_y

    LHS6_xcoord = zeta11_x + zeta21_y
    LHS6_ycoord = zeta12_x + zeta22_y

    lossa = (mse_cost_function(LHS1_xcoord + LHS2_xcoord + LHS3_xcoord + LHS4_xcoord + LHS5_xcoord + LHS6_xcoord ,RHS_xcoord)
        + mse_cost_function(LHS1_ycoord + LHS2_ycoord + LHS3_ycoord + LHS4_ycoord + LHS5_ycoord + LHS6_ycoord ,RHS_ycoord))
    
    #(2.1b)
    div = u1_x + u2_y
    zero = torch.zeros_like(u1).to(device)

    lossb = mse_cost_function(div, zero)

    #(2.1c)
    lossc = mse_cost_function( phi_x*u1 + phi*u1_x + phi_y*u2 + phi*u2_y + mobility*(m_D_xx + m_D_yy), zero)

    #(2.1d)

    lossd = mse_cost_function(-sigma*epsilon_thick*(phi_xx + phi_yy) + sigma/epsilon_thick *(phi**2 -1)*phi, zero)      

    
    # Combine
    loss = lossa + lossb + lossc + lossd
    
    return loss

