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
        self.output_layer_u1N = nn.BatchNorm1d(1)
        
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
        self.output_layer_u2N = nn.BatchNorm1d(1)
        
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
        self.output_layer_PN = nn.BatchNorm1d(1)
        
        self.hidden_layer1_rho = nn.Linear(3,N) # number of input variables = 3, output variables = N
        self.hidden_layer1_rhoN = nn.BatchNorm1d(N)
        self.hidden_layer2_rho = nn.Linear(N,N)
        self.hidden_layer2_rhoN = nn.BatchNorm1d(N)
        #self.hidden_layer3_rho = nn.Linear(N,N)
        #self.hidden_layer4_rho = nn.Linear(N,N)
        #self.hidden_layer5_rho = nn.Linear(N,N)
        #self.hidden_layer6_rho = nn.Linear(N,N)
        #self.hidden_layer7_rho = nn.Linear(N,N)
        #self.hidden_layer8_rho = nn.Linear(N,1)
        self.output_layer_rho = nn.Linear(N,1)
        self.output_layer_rhoN = nn.BatchNorm1d(1)

        #Set Adam algorithm with output of the neural network processing.
        self.optimizer = torch.optim.Adam(self.parameters())
        
        self.mu_1 = 10
        self.mu_2 = 1
        
        self.rho_1 = 1000
        self.rho_2 = 100
        
        self.sigma = 24.5
        self.g = .98
        
        self.Re = 35
        
        self.x1_l = 0
        self.x1_u = 1
        self.x2_l = 0
        self.x2_u = 2
        self.t_l = 0
    
        
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
        output_u1 = self.output_layer_u1N(output_u1)
        
        
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
        output_u2 = self.output_layer_u2N(output_u2)
        
        
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
        output_P = self.output_layer_PN(output_P)
        
        
        # s = nn.Softmax(dim=1)
        layer_rho_out = torch.sigmoid(self.hidden_layer1_rho(inputs))
        layer_rho_out = self.hidden_layer1_rhoN(layer_rho_out)
        layer_rho_out = torch.sigmoid(self.hidden_layer2_rho(layer_rho_out))
        layer_rho_out = self.hidden_layer2_rhoN(layer_rho_out)
        #layer_rho_out = torch.sigmoid(self.hidden_layer3_rho(layer_rho_out))
        #layer_rho_out = torch.sigmoid(self.hidden_layer4_rho(layer_rho_out))
        #layer_rho_out = torch.sigmoid(self.hidden_layer5_rho(layer_rho_out))
        output_rho = torch.sigmoid(self.output_layer_rho(layer_rho_out))
        #output_rho = self.output_layer_rhoN(output_rho)#sigmoid on the last layer
         
        output_rho = -900 * output_rho + 1000 * torch.ones_like(output_rho).to(device)
        
        ## For regression, no activation is used in output layer
        return output_u1, output_u2, output_P, output_rho
        
    
    def mu(self, x, y, t):
        u1, u2, P, rho = self(x,y,t)
        #rho = rho[:,:1] + rho[:,-1:]
        mu = self.mu_1 * (rho - self.rho_2)/(self.rho_1 - self.rho_2) + self.mu_2 * (rho - self.rho_1)/(self.rho_2 - self.rho_1)
        return mu
            
# Output from 'forward' function is u=(u1,u2), P, p, m
# Result of the above class: 
# transformation x,y, t -> u=(u1,u2), P, p, m
# -> torch.sigmoid(u)=torch.sigmoid((u1,u2)), torch.sigmoid(P), torch.sigmoid(p), torch.sigmoid(m) through each hiddeon layer


# In[ ]:




