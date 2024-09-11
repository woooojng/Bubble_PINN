#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set the coefficient to differentiate the "Gamma Moving Boundary" from the inner domain.
# This coefficient will be multiplied to the MovingBoundary MSE term out of total MSE in the end.

def bubble_area(net, t):
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u
    
    Domain_points = 2000
    
    #Pick IC/NSpde Condition Training Random Points in Numpy
    x_c = np.random.uniform(low= x_l, high=x_u, size=(Domain_points, 1)) 
    y_c = np.random.uniform(low= y_l, high=y_u, size=(Domain_points, 1)) 
    
    #Move to pytorch tensors
    x_c = Variable(torch.from_numpy(x_c).float(), requires_grad=True).to(device)
    y_c = Variable(torch.from_numpy(y_c).float(), requires_grad=True).to(device)
    
    #t = np.linspace(0,final_time,10)
    #center_diff_sum = torch.tensor([0])
    omega2_area_ratio_tensor = torch.ones_like(t)
    
    for i in range(0,t.size(dim=1)):
        #Pick IC Training t starting points to make tensor
        t = Variable(t[i] * torch.ones_like(x_c), requires_grad=True).to(device)
        u1, u2, P, rho = net(x_c, y_c, t)
    
        y_unit = torch.where(rho <101, 1, 0)
        r1_area = np.pi*.25*.25
        
        #Compute the denomenator =int_{Omega2} 1 dx
        omega2_area = (y_u - y_l)*(x_u - x_l)/Domain_points *torch.sum(y_unit)
        omega2_area = torch.clamp(omega2_area, min=.002, max=None)
        omega2_area_ratio_tensor[i] = torch.clamp(r1_area/omega2_area, min=10/9, max=5) #max~100
        
   
    return 9/10 *omega2_area_ratio_tensor

def circularity(net, x, y, t):
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u
    
    Domain_collocation = 1000
    
    #Pick IC/NSpde Condition Training Random Points in Numpy
    x_c = np.random.uniform(low= x_l, high=x_u, size=(Domain_collocation, 1)) 
    y_c = np.random.uniform(low= y_l, high=y_u, size=(Domain_collocation, 1)) 
    
    #Move to pytorch tensors
    x_c = Variable(torch.from_numpy(x_c).float(), requires_grad=True).to(device)
    y_c = Variable(torch.from_numpy(y_c).float(), requires_grad=True).to(device)
    
    #t = np.linspace(0,final_time,spatial_discretization)
    
    #Pick IC Training t starting points to make tensor
    
    u1_MvBdry, u2_MvBdry, P_MvBdry, p_MvBdry = net(x, y, t)
    indicator = (torch.where(p_MvBdry <net.rho_1, 1, 0)-torch.where(p_MvBdry <=net.rho_2, 1, 0)) 
    C = []
    perimeter_tensor = torch.ones_like(t)
    for i in range(0, t.size(dim=0)):
        
        if indicator[i] ==1 and torch.sum(indicator) < t.size(dim=0)*.01:
            t_c = Variable(t[i] * torch.ones_like(x_c), requires_grad=True).to(device)
        
            #get actual initial condition
            u1, u2, P, rho = net(x_c, y_c, t_c)
   
        
    
            #compute perimeter of the bubble
            bubble_perimeter = ((y_u - y_l)*(x_u - x_l)/Domain_collocation) * torch.sum(torch.where(rho <net.rho_2+1, 1, 0)
                                                                                -torch.where(rho<=net.rho_2, 1, 0))
            bubble_perimeter = torch.clamp(bubble_perimeter, min = .00000001, max = 0.0505)
            ###??? How to remain only moving bdry pts through domain?
            perimeter_tensor[i] = bubble_perimeter
    
    return perimeter_tensor #Start from .2

def coefficientMvBdry(net, x, y, t):
    u1_MvBdry, u2_MvBdry, P_MvBdry, p_MvBdry = net(x, y, t)
    #p_MvBdry = p_MvBdry[:,:1] + p_MvBdry[:,-1:]
    '''
    # Compute \nabla p
    p_x_MvBdry = torch.autograd.grad(p_MvBdry.sum(), x,create_graph=True)[0].to(device) #p_x
    p_y_MvBdry = torch.autograd.grad(p_MvBdry.sum(), y,create_graph=True)[0].to(device) #p_y
    
    n1 = p_x_MvBdry/(p_x_MvBdry**2 + p_y_MvBdry**2)**.5
    n2 = p_y_MvBdry/(p_x_MvBdry**2 + p_y_MvBdry**2)**.5
    
    # Compute \nabla P
    P_x_MvBdry = torch.autograd.grad(P_MvBdry.sum(), x,create_graph=True)[0] # Compute P_x
    P_y_MvBdry = torch.autograd.grad(P_MvBdry.sum(), y,create_graph=True)[0] #P_y
    
    #Define normal vector n = \nabla p / |\nabla p|
    gradp_MvBdry = torch.cat((p_x_MvBdry, p_y_MvBdry), 1) #.clamp(min=10**-5,max=10**5)
    #Define normal vector n = \nabla P / |\nabla P|
    gradP_MvBdry = torch.cat((P_x_MvBdry, P_y_MvBdry), 1) 
    
    gradp_norm_MvBdry = torch.cdist(gradp_MvBdry, torch.zeros(1, 2).to(device), p=2)
    gradP_norm_MvBdry = torch.cdist(gradP_MvBdry, torch.zeros(1, 2).to(device), p=2)
    
    
    #Begin computing jump in u
    #Compute slightly shifted values based on n
    small = .03 #Keep in mind the size of this
    x_shift_inner = x - small * n1
    x_shift_outer = x + small * n1
    
    y_shift_inner = y - small * n2
    y_shift_outer = y + small * n2
    
    #use normal vector to check continuity across the boundary
    u1_inner, u2_inner, P_inner, p_inner = net(x_shift_inner, y_shift_inner, t)
    u1_outer, u2_outer, P_outer, p_outer = net(x_shift_outer, y_shift_outer, t)
    
    #Define normal vector n = \nabla p / |\nabla p|
    p_jump_norm = torch.cdist(p_outer - p_inner, torch.zeros(1, 1).to(device), p=2)
    
    
    
    MvBdryCoefficient = (p_jump_norm)/(.00001 + (gradp_norm_MvBdry)) #Highlighting ver. for the MvBdry
    #1/((gradp_norm_MvBdry)+ torch.ones_like(gradp_norm_MvBdry)) #Ignoring ver. for the MvBdry
    '''
    
    
    MvBdryCoefficient = torch.div(torch.where(p_MvBdry <net.rho_1, 1, 0)-torch.where(p_MvBdry <=net.rho_2, 1, 0), circularity(net,x,y,t)) 
    #(torch.where(p_MvBdry <net.rho_1, 1, 0)-torch.where(p_MvBdry <=net.rho_2, 1, 0)) /circularity(net,x,y,t)
    
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
    small = .03 #Keep in mind the size of this
    x_shift_inner = x - small * n1
    x_shift_outer = x + small * n1
    
    y_shift_inner = y - small * n2
    y_shift_outer = y + small * n2
    
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
    jump_u1 = u1_inner-u1_outer
    jump_u2 = u2_inner-u2_outer
    
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
    Jump_loss = (mse_cost_function(coefficientMvBdry(net, x, y, t) * jump_u1, zero) + mse_cost_function(coefficientMvBdry(net, x, y, t) * jump_u2, zero))
    SurfaceTension_loss = (mse_cost_function(coefficientMvBdry(net, x, y, t) * SurfaceTension_error_xcoord, zero) + mse_cost_function(coefficientMvBdry(net, x, y, t) * SurfaceTension_error_ycoord, zero))
    
    #Jump_loss = mse_cost_function(jump_u1, zero) + mse_cost_function(jump_u2, zero)
    #SurfaceTension_loss = mse_cost_function(SurfaceTension_error_xcoord, zero) + mse_cost_function(SurfaceTension_error_ycoord, zero)
    
    #Normalize
    #Jump_loss = (Jump_loss - Jump_loss.mean())/(1+Jump_loss.std())
    #SurfaceTension_loss = (SurfaceTension_loss - SurfaceTension_loss.mean())/(1+SurfaceTension_loss.std())
    
    # Final Moving Boundary Loss Function in total
    MovingBdry_loss = Jump_loss + SurfaceTension_loss
    
    return Jump_loss, SurfaceTension_loss


    
    

