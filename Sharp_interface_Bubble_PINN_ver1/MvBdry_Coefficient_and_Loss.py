#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set the coefficient to differentiate the "Gamma Moving Boundary" from the inner domain.
# This coefficient will be multiplied to the MovingBoundary MSE term out of total MSE in the end.

def coefficientMvBdry(net, x, y, t):
    u1_MvBdry, u2_MvBdry, P_MvBdry, p_MvBdry = net(x, y, t)
    #p_MvBdry = p_MvBdry[:,:1] + p_MvBdry[:,-1:]
    
    # Compute \nabla p
    p_x_MvBdry = torch.autograd.grad(p_MvBdry.sum(), x,create_graph=True)[0].to(device) #p_x
    p_y_MvBdry = torch.autograd.grad(p_MvBdry.sum(), y,create_graph=True)[0].to(device) #p_y
    
    #Define normal vector n = \nabla p / |\nabla p|
    gradp_MvBdry = torch.cat((p_x_MvBdry, p_y_MvBdry), 1) #.clamp(min=10**-5,max=10**5)
    
    gradp_norm_MvBdry = torch.cdist(gradp_MvBdry, torch.zeros(1, 2).to(device), p=2)
    
    
    #Compute Tangent Hyperbolic function
    #to recognize only moving boundary points through the domain.
    
    #temp = gradp_norm_MvBdry/torch.exp(3* torch.ones_like(gradp_norm_MvBdry.to(device)))
    #MvBdryCoefficient = (torch.tanh(temp))**2
    MvBdryCoefficient = .01 * (gradp_norm_MvBdry) #Highlighting ver. for the MvBdry
    #1/((gradp_norm_MvBdry)+ torch.ones_like(gradp_norm_MvBdry)) #Ignoring ver. for the MvBdry
    
    return MvBdryCoefficient

def lossMvBdry(net, x, y, t):
    mse_cost_function = torch.nn.MSELoss() #reduction='sum'
    
    #Compute values at (x,y,t)
    u1_MvBdry, u2_MvBdry, P_MvBdry, p_MvBdry = net(x, y, t)
    #p_MvBdry = p_MvBdry[:,:1] + p_MvBdry[:,-1:]
    m = net.mu(x,y,t)
    
    #Compute normal vector using p
    p_x_MvBdry = torch.autograd.grad(p_MvBdry.sum(), x, create_graph=True)[0].to(device) # Compute p_x
    p_y_MvBdry = torch.autograd.grad(p_MvBdry.sum(), y, create_graph=True)[0].to(device) #p_y
    n1 = p_x_MvBdry/(p_x_MvBdry**2 + p_y_MvBdry**2)**.5
    n2 = p_y_MvBdry/(p_x_MvBdry**2 + p_y_MvBdry**2)**.5
    #gradp_MvBdry = torch.cat((p_x_MvBdry, p_y_MvBdry), 1)
    #n_MvBdry = torch.nn.functional.normalize(gradp_MvBdry, p=2.0, dim=1, eps=1e-12, out=None)
    
    #Begin computing jump in u
    #Compute slightly shifted values based on n
    small = .1 #Keep in mind the size of this
    x_shift_inner = x + small * n1
    x_shift_outer = x - small * n1
    
    y_shift_inner = y + small * n2
    y_shift_outer = y - small * n2
    
    #use normal vector to check continuity across the boundary
    u1_inner, u2_inner, P_inner, p_inner = net(x_shift_inner, y_shift_inner, t)
    #p_inner = p_inner[:,:1] + p_inner[:,-1:]
    m_inner = net.mu(x_shift_inner, y_shift_inner, t)
    
    u1_outer, u2_outer, P_outer, p_outer = net(x_shift_outer, y_shift_outer, t)
    #p_outer = p_outer[:,:1] + p_outer[:,-1:]
    
    m_outer = net.mu(x_shift_outer, y_shift_outer, t)
    
    #Computer inner and outer derivatives
    u1_x_inner = torch.autograd.grad(u1_inner.sum(), x_shift_inner, create_graph=True)[0].to(device)
    u1_y_inner = torch.autograd.grad(u1_inner.sum(), y_shift_inner, create_graph=True)[0].to(device)
    
    u2_x_inner = torch.autograd.grad(u2_inner.sum(), x_shift_inner, create_graph=True)[0].to(device)
    u2_y_inner = torch.autograd.grad(u2_inner.sum(), y_shift_inner, create_graph=True)[0].to(device)
    
    u1_x_outer = torch.autograd.grad(u1_outer.sum(), x_shift_outer, create_graph=True)[0].to(device)
    u1_y_outer = torch.autograd.grad(u1_outer.sum(), y_shift_outer, create_graph=True)[0].to(device)
    
    u2_x_outer = torch.autograd.grad(u2_outer.sum(), x_shift_outer, create_graph=True)[0].to(device)
    u2_y_outer = torch.autograd.grad(u2_outer.sum(), y_shift_outer, create_graph=True)[0].to(device)
    
    #Compute predicted jump across boudnary
    jump_u1 = torch.cdist(u1_inner-u1_outer, torch.zeros(1, 1).to(device), p=1)
    jump_u2 = torch.cdist(u2_inner-u2_outer, torch.zeros(1, 1).to(device), p=1)
    
    #Completed computing jump in u
    
    #Begin computing surface tension
    #Now, lets define/initialize normal vector n, curvature k = \nabla \cdot n and tangential vector tau.   
    n1_x_MvBdry = torch.autograd.grad(n1.sum(),x, create_graph=True)[0].to(device) 
    n2_y_MvBdry = torch.autograd.grad(n2.sum(),y, create_graph=True)[0].to(device)
    k_MvBdry = n1_x_MvBdry + n2_y_MvBdry
    
    #Compute SurfaceTension - I assumed surface tension coefficient as \sigma = 24.5
    SurfaceTension_term1_xcoord_inner = -p_inner + m_inner * (u1_x_inner)
    SurfaceTension_term2_xcoord_inner = .5 * m_inner * (u1_y_inner + u2_x_inner)
    SurfaceTension_term1_ycoord_inner = .5 * m_inner * (u1_y_inner + u2_x_inner)
    SurfaceTension_term2_ycoord_inner = -p_inner + m_inner * u2_y_inner
    
    SurfaceTension_term1_xcoord_outer = -p_outer + m_outer * (u1_x_outer)
    SurfaceTension_term2_xcoord_outer = .5 * m_outer * (u1_y_outer + u2_x_outer)
    SurfaceTension_term1_ycoord_outer = .5 * m_outer * (u1_y_outer + u2_x_outer)
    SurfaceTension_term2_ycoord_outer = -p_outer + m_outer * u2_y_outer
    
    SurfaceTension_Jump_xcoord = (SurfaceTension_term1_xcoord_outer - SurfaceTension_term1_xcoord_inner)*n1 + (SurfaceTension_term2_xcoord_outer - SurfaceTension_term2_xcoord_inner)*n2
    SurfaceTension_Jump_ycoord = (SurfaceTension_term1_ycoord_outer - SurfaceTension_term1_ycoord_inner)*n1 + (SurfaceTension_term2_ycoord_outer - SurfaceTension_term2_ycoord_inner)*n2
    
    SurfaceTension_RHS_xcoord = .5 * net.sigma * k_MvBdry * n1
    SurfaceTension_RHS_ycoord = .5 * net.sigma * k_MvBdry * n2
    
    SurfaceTension_error_xcoord = SurfaceTension_Jump_xcoord - SurfaceTension_RHS_xcoord
    SurfaceTension_error_ycoord = SurfaceTension_Jump_ycoord - SurfaceTension_RHS_ycoord

    #Surface Tension computation completed    

    # Define MSE Function on Moving Boundary, we use custom weightings so choose no reduction
    zero = torch.zeros_like(x)
    Jump_loss = mse_cost_function(coefficientMvBdry(net, x, y, t) * jump_u1, zero) + mse_cost_function(coefficientMvBdry(net, x, y, t) * jump_u2, zero)
    SurfaceTension_loss = mse_cost_function(coefficientMvBdry(net, x, y, t) * SurfaceTension_error_xcoord, zero) + mse_cost_function(coefficientMvBdry(net, x, y, t) * SurfaceTension_error_ycoord, zero)
    
    #Normalize
    #Jump_loss = (Jump_loss - Jump_loss.mean())/(1+Jump_loss.std())
    #SurfaceTension_loss = (SurfaceTension_loss - SurfaceTension_loss.mean())/(1+SurfaceTension_loss.std())
    
    # Final Moving Boundary Loss Function in total
    MovingBdry_loss = Jump_loss + SurfaceTension_loss
    
    return MovingBdry_loss


    
    

