#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lossBdry(net, x, y, t, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry):
    mse_cost_function = torch.nn.MSELoss(reduction='mean')
    zero = torch.zeros_like(x)
    
    
    # Define the variables on 4 boundaries of outer squre
    u1_lower, u2_lower, _, phi_lower, m_D_lower = net(x, y_l_Bdry, t)
    u1_upper, u2_upper, _, phi_upper, m_D_upper = net(x, y_u_Bdry, t)
    u1_left, u2_left, _, phi_left, m_D_left = net(x_l_Bdry, y, t)
    u1_right, u2_right, _, phi_right, m_D_right = net(x_u_Bdry, y, t)
    
    ## Prep for Free-Slip Boundary Condition
    
    #Compute left wall derivatives
    u1_y_left = torch.autograd.grad(u1_left.sum(), y, create_graph=True)[0].to(device) #Compute u1_y on left wall
    u2_x_left = torch.autograd.grad(u2_left.sum(), x_l_Bdry, create_graph=True)[0].to(device) #Compute u2_x on left wall
    
    #Compute right wall derivatives
    u1_y_right = torch.autograd.grad(u1_right.sum(), y, create_graph=True)[0].to(device) #Compute u1_y on right wall
    u2_x_right = torch.autograd.grad(u2_right.sum(), x_u_Bdry, create_graph=True)[0].to(device) #Compute u2_x on right wall

    
    ##### Free-Slip Boundary Condition on the vertical wall boundaries
    #u \cdot n = 0 on left and right
    un_left_loss = mse_cost_function(u1_left, zero)
    un_right_loss = mse_cost_function(u1_right, zero)
    
    #tau Du n = 0
    complex_left_loss = mse_cost_function(u1_y_left + u2_x_left, zero)
    complex_right_loss = mse_cost_function(u1_y_right + u2_x_right, zero)
    
    ##### Top and Bottom No Slip Loss
    Top_Loss = mse_cost_function(u1_upper, zero) + mse_cost_function(u2_upper, zero)
    Bot_Loss = mse_cost_function(u1_lower, zero) + mse_cost_function(u2_lower, zero)
    
    
    #Left and Right Loss
    Left_Loss = un_left_loss + complex_left_loss 
    Right_Loss = un_right_loss + complex_right_loss
        
    # Final Boundary Loss Function in total
    boundary_loss_u = Top_Loss + Bot_Loss + Left_Loss + Right_Loss
    
    #boundary_loss = (boundary_loss-boundary_loss.mean())/(1+boundary_loss.std())

    ##### Neumann bdry condition of phi, mu on left/right with normal vector n := (+-1, 0)

    # Compute \nabla phi \cdot n, \nabla m_D \cdot n on left/right
    phi_x_left = torch.autograd.grad(phi_left.sum(), x_l_Bdry,create_graph=True)[0].to(device)
    phi_x_right = torch.autograd.grad(phi_right.sum(), x_u_Bdry,create_graph=True)[0].to(device)
    m_D_x_left = torch.autograd.grad(m_D_left.sum(), x_l_Bdry,create_graph=True)[0].to(device)
    m_D_x_right = torch.autograd.grad(m_D_right.sum(), x_u_Bdry,create_graph=True)[0].to(device)
    
    Left_Loss_phi = mse_cost_function(phi_x_left, zero)
    Right_Loss_phi = mse_cost_function(phi_x_right, zero)
    Left_Loss_mu = mse_cost_function(m_D_x_left, zero)
    Right_Loss_mu = mse_cost_function(m_D_x_right, zero)
    
    ##### Neumann bdry condition of phi, mu on top/bottom with normal vector n := (0, +-1)

    # Compute \nabla phi \cdot n, \nabla m_D \cdot n on top/bottom
    phi_y_upper = torch.autograd.grad(phi_upper.sum(), y_u_Bdry,create_graph=True)[0].to(device)
    phi_y_lower = torch.autograd.grad(phi_lower.sum(), y_l_Bdry,create_graph=True)[0].to(device)
    m_D_y_upper = torch.autograd.grad(m_D_upper.sum(), y_u_Bdry,create_graph=True)[0].to(device)
    m_D_y_lower = torch.autograd.grad(m_D_lower.sum(), y_l_Bdry,create_graph=True)[0].to(device)    
    
    Top_Loss_phi = mse_cost_function(phi_y_upper, zero)
    Bot_Loss_phi = mse_cost_function(phi_y_lower, zero)
    Top_Loss_mu = mse_cost_function(m_D_y_upper, zero)
    Bot_Loss_mu = mse_cost_function(m_D_y_lower, zero)

    boundary_loss_phimu = Left_Loss_phi + Right_Loss_phi + Left_Loss_mu + Right_Loss_mu + Top_Loss_phi + Bot_Loss_phi + Top_Loss_mu + Bot_Loss_mu
    
    return boundary_loss_u + boundary_loss_phimu


