#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
    	
        super(Net, self).__init__()

        N=10 #500
        self.hidden_layer1_u1 = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_u1N = nn.BatchNorm1d(N)
        self.hidden_layer2_u1 = nn.Linear(N,N)
        self.hidden_layer2_u1N = nn.BatchNorm1d(N)
        #self.hidden_layer3_u1 = nn.Linear(N,N)
        #self.hidden_layer4_u1 = nn.Linear(N,N)
        #self.hidden_layer5_u1 = nn.Linear(N,N)
        #self.hidden_layer6_u1 = nn.Linear(N,N)
        #self.hidden_layer7_u1 = nn.Linear(N,N)
        #self.hidden_layer8_u1 = nn.Linear(N,N)
        #self.hidden_layer9_u1 = nn.Linear(N,N)
        #self.hidden_layer10_u1 = nn.Linear(N,N)
        self.output_layer_u1 = nn.Linear(N,1)
        #self.output_layer_u1N = nn.BatchNorm1d(1)
        
        self.hidden_layer1_u2 = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_u2N = nn.BatchNorm1d(N)
        self.hidden_layer2_u2 = nn.Linear(N,N)
        self.hidden_layer2_u2N = nn.BatchNorm1d(N)
        #self.hidden_layer3_u2 = nn.Linear(N,N)
        #self.hidden_layer4_u2 = nn.Linear(N,N)
        #self.hidden_layer5_u2 = nn.Linear(N,N)
        #self.hidden_layer6_u2 = nn.Linear(N,N)
        #self.hidden_layer7_u2 = nn.Linear(N,N)
        #self.hidden_layer8_u2 = nn.Linear(N,N)
        #self.hidden_layer9_u2 = nn.Linear(N,N)
        #self.hidden_layer10_u2 = nn.Linear(N,N)
        self.output_layer_u2 = nn.Linear(N,1)
        #self.output_layer_u2N = nn.BatchNorm1d(1)
        
        self.hidden_layer1_P = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_PN = nn.BatchNorm1d(N)
        self.hidden_layer2_P = nn.Linear(N,N)
        self.hidden_layer2_PN = nn.BatchNorm1d(N)
        #self.hidden_layer3_P = nn.Linear(N,N)
        #self.hidden_layer4_P = nn.Linear(N,N)
        #self.hidden_layer5_P = nn.Linear(N,N)
        #self.hidden_layer6_P = nn.Linear(N,N)
        #self.hidden_layer7_P = nn.Linear(N,N)
        #self.hidden_layer8_P = nn.Linear(N,N)
        self.output_layer_P = nn.Linear(N,1)
        #self.output_layer_PN = nn.BatchNorm1d(1)
        
        self.hidden_layer1_phi = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_phiN = nn.BatchNorm1d(N)
        self.hidden_layer2_phi = nn.Linear(N,N)
        self.hidden_layer2_phiN = nn.BatchNorm1d(N)
        #self.hidden_layer3_phi = nn.Linear(N,N)
        #self.hidden_layer4_phi = nn.Linear(N,N)
        #self.hidden_layer5_phi = nn.Linear(N,N)
        #self.hidden_layer6_phi = nn.Linear(N,N)
        #self.hidden_layer7_phi = nn.Linear(N,N)
        #self.hidden_layer8_phi = nn.Linear(N,1)
        self.output_layer_phi = nn.Linear(N,1)
        #self.output_layer_phiN = nn.BatchNorm1d(1)

        self.hidden_layer1_m_D = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_m_DN = nn.BatchNorm1d(N)
        self.hidden_layer2_m_D = nn.Linear(N,N)
        self.hidden_layer2_m_DN = nn.BatchNorm1d(N)
        #self.hidden_layer3_m_D = nn.Linear(N,N)
        #self.hidden_layer4_m_D = nn.Linear(N,N)
        #self.hidden_layer5_m_D = nn.Linear(N,N)
        #self.hidden_layer6_m_D = nn.Linear(N,N)
        #self.hidden_layer7_m_D = nn.Linear(N,N)
        #self.hidden_layer8_m_D = nn.Linear(N,N)
        self.output_layer_m_D = nn.Linear(N,1)
        #self.output_layer_m_DN = nn.BatchNorm1d(1)

        #Set Adam algorithm with output of the neural network processing.
        self.optimizer = torch.optim.Adam(self.parameters())
        
        self.mu_1 = 10
        self.mu_2 = 1
        
        self.rho_1 = 1000
        self.rho_2 = 100
        
        self.sigma_DA = 24.5
        self.g = .98
        
        self.Re = 35
        
        self.x1_l = 0
        self.x1_u = 1
        self.x2_l = 0
        self.x2_u = 2
        self.t_l = 0

        self.epsilon_thick = .1 #interface thickness parameter
        self.mobility = .1 #mobility parameter
        self.sigma = (3*self.sigma_DA)/(2*np.sqrt(2)) #rescaling of the droplet-ambient surface tension sigma_DA with 2*sqrt(2)*sigma=3*sigma_DA
        
    
        
    def forward(self, x, y, t):
        inputs = torch.cat([x,y,t],axis=1) # Merged three single-column arrays into a single array with two columns.
        #flatten = nn.Flatten()
        #inputs = flatten(inputs)
        
        layer_u1_out = torch.sigmoid(self.hidden_layer1_u1(inputs))
        layer_u1_out = self.hidden_layer1_u1N(layer_u1_out)
        layer_u1_out = torch.sigmoid(self.hidden_layer2_u1(layer_u1_out))
        layer_u1_out = self.hidden_layer2_u1N(layer_u1_out)
        #layer_u1_out = torch.sigmoid(self.hidden_layer3_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer4_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer5_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer6_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer7_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer8_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer9_u1(layer_u1_out))
        #layer_u1_out = torch.sigmoid(self.hidden_layer10_u1(layer_u1_out))
        output_u1 = self.output_layer_u1(layer_u1_out)
        #output_u1 = self.output_layer_u1N(output_u1)
        
        
        layer_u2_out = torch.sigmoid(self.hidden_layer1_u2(inputs))
        layer_u2_out = self.hidden_layer1_u2N(layer_u2_out)
        layer_u2_out = torch.sigmoid(self.hidden_layer2_u2(layer_u2_out))
        layer_u2_out = self.hidden_layer2_u2N(layer_u2_out)
        #layer_u2_out = torch.sigmoid(self.hidden_layer3_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer4_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer5_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer6_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer7_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer8_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer9_u2(layer_u2_out))
        #layer_u2_out = torch.sigmoid(self.hidden_layer10_u2(layer_u2_out))
        output_u2 = self.output_layer_u2(layer_u2_out)
        #output_u2 = self.output_layer_u2N(output_u2)
        
        
        layer_P_out = torch.sigmoid(self.hidden_layer1_P(inputs))
        layer_P_out = self.hidden_layer1_PN(layer_P_out)
        layer_P_out = torch.sigmoid(self.hidden_layer2_P(layer_P_out))
        layer_P_out = self.hidden_layer2_PN(layer_P_out)
        #layer_P_out = torch.sigmoid(self.hidden_layer3_P(layer_P_out))
        #layer_P_out = torch.sigmoid(self.hidden_layer4_P(layer_P_out))
        #layer_P_out = torch.sigmoid(self.hidden_layer5_P(layer_P_out))
        #layer_P_out = torch.sigmoid(self.hidden_layer6_P(layer_P_out))
        #layer_P_out = torch.sigmoid(self.hidden_layer7_P(layer_P_out))
        #layer_P_out = torch.sigmoid(self.hidden_layer8_P(layer_P_out))
        output_P = self.output_layer_P(layer_P_out)
        #output_P = self.output_layer_PN(output_P)
        
        
        # s = nn.Softmax(dim=1)
        layer_phi_out = torch.sigmoid(self.hidden_layer1_phi(inputs))
        layer_phi_out = self.hidden_layer1_phiN(layer_phi_out)
        layer_phi_out = torch.sigmoid(self.hidden_layer2_phi(layer_phi_out))
        layer_phi_out = self.hidden_layer2_phiN(layer_phi_out)
        #layer_phi_out = torch.sigmoid(self.hidden_layer3_phi(layer_phi_out))
        #layer_phi_out = torch.sigmoid(self.hidden_layer4_phi(layer_phi_out))
        #layer_phi_out = torch.sigmoid(self.hidden_layer5_phi(layer_phi_out))
        output_phi = torch.tanh(self.output_layer_phi(layer_phi_out))
        #output_phi = self.output_layer_phiN(output_phi)#sigmoid on the last layer
         
        #output_rho = -900 * output_rho + 1000 * torch.ones_like(output_rho).to(device)

        layer_m_D_out = torch.sigmoid(self.hidden_layer1_m_D(inputs))
        layer_m_D_out = self.hidden_layer1_m_DN(layer_m_D_out)
        layer_m_D_out = torch.sigmoid(self.hidden_layer2_m_D(layer_m_D_out))
        layer_m_D_out = self.hidden_layer2_m_DN(layer_m_D_out)
        #layer_m_D_out = torch.sigmoid(self.hidden_layer3_m_D(layer_m_D_out))
        #layer_m_D_out = torch.sigmoid(self.hidden_layer4_m_D(layer_m_D_out))
        #layer_m_D_out = torch.sigmoid(self.hidden_layer5_m_D(layer_m_D_out))
        #layer_m_D_out = torch.sigmoid(self.hidden_layer6_m_D(layer_m_D_out))
        #layer_m_D_out = torch.sigmoid(self.hidden_layer7_m_D(layer_m_D_out))
        #layer_m_D_out = torch.sigmoid(self.hidden_layer8_m_D(layer_m_D_out))
        output_m_D = self.output_layer_m_D(layer_m_D_out)
        #output_m_D = self.output_layer_m_DN(output_m_D)
        
        ## For regression, no activation is used in output layer
        return output_u1, output_u2, output_P, output_phi, output_m_D
        
    
    def mu(self, x, y, t): #viscoscity
        u1, u2, P, phi, m_D = self(x,y,t)
        rho = self.rho(x,y,t)
        mu = self.mu_1 * (rho - self.rho_2)/(self.rho_1 - self.rho_2) + self.mu_2 * (rho - self.rho_1)/(self.rho_2 - self.rho_1)
        return mu

    def rho(self, x, y, t): #density
        u1, u2, P, phi, _ = self(x,y,t)
        rho = (1 - phi)/2 *self.rho_1 + (1 + phi)/2 *self.rho_2
        return rho

    def eta(self, x, y, t): #viscoscity mixture
        u1, u2, P, phi, _ = self(x,y,t)
        eta = torch.exp(((1 - phi) * torch.log(10))/((1 + phi)*100 + (1 - phi)))
        return eta

    def J(self, x, y, t): #
        u1, u2, P, phi, m_D = self(x,y,t)

        m_D_x = torch.autograd.grad(m_D.sum(), x,create_graph=True)[0] #Compute m_x
        m_D_y = torch.autograd.grad(m_D.sum(), y,create_graph=True)[0] #Compute m_y
    
        J1 = self.mobility *(self.rho_1 - self.rho_2)/2*m_D_x
        J2 = self.mobility *(self.rho_1 - self.rho_2)/2*m_D_y
        return J1, J2

    def tau(self, x, y, t): #viscous stress; diffrent from tangential vector on Mvbdry
        u1, u2, P, phi, _ = self(x,y,t)

        #Compute Derivatives
        u1_x = torch.autograd.grad(u1.sum(), x,create_graph=True)[0] #Compute u1_x
        u1_y = torch.autograd.grad(u1.sum(), y,create_graph=True)[0] #Compute u1_y
    
        u2_x = torch.autograd.grad(u2.sum(), x,create_graph=True)[0] #Compute u2_x
        u2_y = torch.autograd.grad(u2.sum(), y,create_graph=True)[0] #Compute u2_y
        tau11 = 2 *u1_x
        tau12 = u1_y + u2_x
        tau21 = u1_y + u2_x
        tau22 = 2 *u2_y
        
        return tau11, tau12, tau21, tau22
    
    def zeta(self, x, y, t): #
        u1, u2, P, phi, m = self(x,y,t)
        psi = self.psi(x,y,t)

        #Compute Derivatives
        phi_x = torch.autograd.grad(phi.sum(), x,create_graph=True)[0] #Compute phi_x
        phi_y = torch.autograd.grad(phi.sum(), y,create_graph=True)[0] #Compute phi_y
    
        zeta11 = (- self.sigma * self.epsilon_thick *phi_x**2 +
            self.sigma*self.epsilon_thick /2 *(phi_x**2 + phi_y**2)+self.sigma / self.epsilon_thick*psi)
        zeta12 = (- self.sigma * self.epsilon_thick *phi_x*phi_y)
        zeta21 = (- self.sigma * self.epsilon_thick *phi_x*phi_y)
        zeta22 = (- self.sigma * self.epsilon_thick *phi_y**2+
            self.sigma*self.epsilon_thick /2 *(phi_x**2 + phi_y**2)+self.sigma / self.epsilon_thick*psi)
        return zeta11, zeta12, zeta21, zeta22
    
    def psi(self, x, y, t): #
        u1, u2, P, phi, m = self(x,y,t)
        psi = 1/4 * (phi**2 - torch.ones_like(phi))**2
        return psi

            




