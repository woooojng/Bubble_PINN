#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#from torch.autograd import Variable

def c_mass_area(net, final_time):
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u
    
    Domain_points = 10
    
    #Pick IC/NSpde Condition Training Random Points in Numpy
    x_c = np.random.uniform(low= x_l, high=x_u, size=(Domain_points, 1)) 
    y_c = np.random.uniform(low= y_l, high=y_u, size=(Domain_points, 1)) 
    
    #Move to pytorch tensors
    x_c = Variable(torch.from_numpy(x_c).float(), requires_grad=True).to(device)
    y_c = Variable(torch.from_numpy(y_c).float(), requires_grad=True).to(device)
    
    t = np.linspace(0,final_time,10)
    center_diff_sum = torch.tensor([0])
    omega2_area_ratio_sum = torch.tensor([0])
    
    for pt_t in t:
        #Pick IC Training t starting points to make tensor
        t = Variable(pt_t * torch.ones_like(x_c), requires_grad=True).to(device)
        u1, u2, P, rho = net(x_c, y_c, t)
    
        y_unit = torch.where(rho <101, 1, 0)
        r1_area = np.pi*.25*.25
        
        #Compute the denomenator =int_{Omega2} 1 dx
        omega2_area = (y_u - y_l)*(x_u - x_l)/Domain_points *torch.sum(y_unit)
        omega2_area = torch.clamp(omega2_area, min=.002, max=r1_area*0.81)
        omega2_area_ratio_sum = omega2_area_ratio_sum + torch.clamp(r1_area/omega2_area, min=10/9, max=5) #max~100
        
        #Compute the nominator =int_{Omega2} x dx
        y_c = torch.where(rho <101, y_c, 0)
        y_omega2_exp = (y_u - y_l)*(x_u - x_l)/Domain_points *torch.sum(y_c)
        
        x_c = torch.where(rho <101, x_c, 0)
        x_omega2_exp = (y_u - y_l)*(x_u - x_l)/Domain_points *torch.sum(x_c)
        
        y_center_of_mass = y_omega2_exp/omega2_area
        x_center_of_mass = x_omega2_exp/omega2_area
        #get actual initial condition
        
        center_of_mass_difference = torch.clamp(torch.abs(y_center_of_mass- (1/6)* pt_t-.5), min=0, max=0.2) + torch.clamp(torch.abs(x_center_of_mass-.5), min=0, max=0.2)
        center_diff_sum = center_diff_sum + center_of_mass_difference
    
   
    return torch.sum(center_diff_sum)/10, omega2_area_ratio_sum/10


def lossNSpde(net, x, y, t, final_time):
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
    #u1_xy = torch.autograd.grad(u1_x.sum(),y,create_graph=True)[0] #Compute u1_xy
    u1_yy = torch.autograd.grad(u1_y.sum(),y,create_graph=True)[0] #Compute u1_yy
    
    u2_xx = torch.autograd.grad(u2_x.sum(),x,create_graph=True)[0] #Compute u2_xx
    #u2_xy = torch.autograd.grad(u2_x.sum(),y,create_graph=True)[0] #Compute u2_xy
    u2_yy = torch.autograd.grad(u2_y.sum(),y,create_graph=True)[0] #Compute u2_yy

    
    #Define normal vector n = \nabla P / |\nabla P|
    gradP = torch.cat((P_x, P_y), 1) #.clamp(min=10**-5,max=10**5)
    gradP_norm_coeff = torch.cdist(gradP, torch.zeros(1, 2).to(device), p=2)
    
    #Define normal vector n = \nabla p / |\nabla p|
    gradp = torch.cat((p_x, p_y), 1) #.clamp(min=10**-5,max=10**5)
    
    gradp_norm = torch.cdist(gradp, torch.zeros(1, 2).to(device), p=2)
    gradp_norm = torch.clamp(gradp_norm, min=.005, max=500)
    gradp_norm_coeff = 1/(gradp_norm+ 1) #torch.clamp(1/(gradp_norm+ 1), min=None, max=10)
    
    center_of_mass, bubble_area_ratio = c_mass_area(net, final_time)
    
    #Set the coefficients to be 1 after training fits the PDE system well
    MvBdryCoefficient = center_of_mass * (bubble_area_ratio - 1) * gradp_norm_coeff
    
    
    
    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    g = 0.98
    #NSpde_xcoord = (p*( u1_t + u1 * u1_x + u2 * u1_y) + P_x - (2 * m_x * u1_x + m_y * u2_x + m_y * u1_y)- m * (2*u1_xx +u2_xy + u1_yy) - p_x * g)
    NSpde_xcoord = (p*( u1_t + u1 * u1_x + u2 * u1_y) + P_x - (2 * m_x * u1_x + m_y * u2_x + m_y * u1_y)- m * (u1_xx + u1_yy) )
                    #- p_x * g)
    #NSpde_ycoord = (p*( u2_t + u1 * u2_x + u2 * u2_y) + P_y - (m_x * u1_y + 2 * m_y * u2_y + m_x * u2_x)- m * (u1_xy +2*u2_yy + u2_xx) - p_y * g)
    NSpde_ycoord = (p*( u2_t + u1 * u2_x + u2 * u2_y) + P_y - (m_x * u1_y + 2 * m_y * u2_y + m_x * u2_x)- m * (u2_yy + u2_xx) + p * g) #p_y * g)
    
    zero_NSpde_xcoord = torch.zeros_like(NSpde_xcoord)
    zero_NSpde_ycoord = torch.zeros_like(NSpde_ycoord)

    # \nabula \cdot u = 0
    NSpde_div = u1_x + u2_y
    zero_NSpde_div = torch.zeros_like(NSpde_div)
    
    # Define MSE Function on inner space
    NSpde_loss = (mse_cost_function(MvBdryCoefficient* torch.sin(np.pi * x)* torch.sin(np.pi /2 * y)*NSpde_xcoord, zero_NSpde_xcoord) + mse_cost_function(MvBdryCoefficient* torch.sin(np.pi * x)* torch.sin(np.pi /2 * y)*NSpde_ycoord, zero_NSpde_ycoord))
    NSpde_div_loss = mse_cost_function(MvBdryCoefficient* torch.sin(np.pi * x)* torch.sin(np.pi /2 * y)*NSpde_div, zero_NSpde_div) 
    
    # Normalize
    #NSpde_loss = (NSpde_loss - NSpde_loss.mean())/(1+NSpde_loss.std())
    #NSpde_div_loss = (NSpde_div_loss - NSpde_div_loss.mean())/(1+NSpde_div_loss.std())

    # Combine
    #NS_loss = (NSpde_loss + NSpde_div_loss)
    '''
    areaLoss = mse_cost_function(bubble_area_ratio* 9/10,torch.ones_like(bubble_area_ratio))
    centerLoss = mse_cost_function(center_of_mass,torch.zeros_like(center_of_mass))
    '''
    return NSpde_loss, NSpde_div_loss #, areaLoss, centerLoss

