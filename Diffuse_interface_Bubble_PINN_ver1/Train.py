#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
import time


#Call model of layers and its forward step
from Forward_with_Layer_Setting import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Call training functions of Loss functions
from NSpde_loss import lossNSpde 
from BoundaryLoss import lossBdry
from InitialConditionLoss import lossIC



def create_network(IC_Only_Train):
    
    net = Net()
    net = net.to(device)

    #Set final times for running training
    time_slices = np.array([.01,.1]) #, .25, .5, 1
    
    #Load Training Points
    x_domain, y_domain, t_zero, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry = twoDimTrainPts(net, Domain_collocation = int(1000), Bdry_collocation = int(100))
    
    start = time.time()

    #Start Training only on IC
    if IC_Only_Train == True:
        print('Training Only on the Initial Condition')
        Create_IC_Parameters(x_domain, y_domain, t_zero, 30000, 10**-3, 'IC_Only.pt', record_loss = 100, print_loss = 1000)
        IC_Done = time.time()
        print('IC Time:\t', IC_Done-start)
        '''
        print('Training Only on the NSpde Condition')
        NSpde_Only_training(net, x_domain, y_domain, t_zero, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, 
                      y_l_Bdry, y_u_Bdry, time_slices, 20000, 10**-3, record_loss = 100, print_loss = 1000)
        torch.save(net.state_dict(), f"NSpde_afterIC_.pt")
        NSpde_Done = time.time()
        print('MvBdry Time:\t', NSpde_Done-start)
        '''
        return 0
        
    time_vec = [0, 0, 0, 0]
    
    
    #attempt to load IC if it exists
    try:
        net.load_state_dict(torch.load("IC_Only.pt"))
    except:
        pass
    '''
    #attempt to load MvBdry if it exists
    try:
        net.load_state_dict(torch.load("NSpde_Only.pt"))
    except:
        pass
    '''
    global epsilon #used to track loss
    epsilon = []
    
    print('Training PDE')
    
    for i in range(4):
        #Set loop to optimize in progressively smaller learning rates
        if i == 0:
            #First loop uses progressively increasing time intervals
            print('Executing Pass 1')
            iterations = 150000
            learning_rate = 10**-3    
        elif i == 1:
            print('Executing Pass 2')
            #time_slices = time_slices[-1]
            iterations = 150000
            learning_rate = 10**-2
        elif i == 2:
            print('Executing Pass 3')
            iterations = 2 #0000
            learning_rate = 5*10**-6
        elif i ==3:
            print('Executing Pass 4')
            iterations = 2 #0000
            learning_rate = 10**-6
        
        training_loop(net, x_domain, y_domain, t_zero, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, 
                      y_l_Bdry, y_u_Bdry, time_slices, iterations, learning_rate, IC_coefficient = 1, record_loss = 100, print_loss = 200)
        torch.save(net.state_dict(), f"NNlayers_Bubble_{i}.pt")
        np.savetxt('epsilon.txt', epsilon)
        time_vec[i] = time.time()

    np.savetxt('epsilon.txt', epsilon)
    
    end = time.time()

    print("Total Time:\t", end-start, '\nPass 1 Time:\t', time_vec[0]-start, '\nPass 2 Time:\t', time_vec[1]-start, '\nPass 3 Time:\t', time_vec[2]-start, '\nPass 4 Time:\t', time_vec[3]-start)


def twoDimTrainPts(net, Domain_collocation, Bdry_collocation):
    #Set of all the recorded xy variables as base data for chasing during training
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u

    #time starts at lower bound 0, ends at upper bouund updated in slices
    t_l = 0

    #Pick IC/NSpde Condition Training Random Points in Numpy
    x_domain = np.random.uniform(low= x_l, high=x_u, size=(Domain_collocation, 1)) 
    y_domain = np.random.uniform(low= y_l, high=y_u, size=(Domain_collocation, 1)) 
    
    #Move to pytorch tensors
    x_domain = Variable(torch.from_numpy(x_domain).float(), requires_grad=True).to(device)
    y_domain = Variable(torch.from_numpy(y_domain).float(), requires_grad=True).to(device)
    
    #Pick IC Training t starting points to make tensor
    t_zero = Variable(torch.zeros_like(x_domain), requires_grad=True).to(device)

    #Pick BC Training Random Points in Numpy
    x_Bdry = np.random.uniform(low=x_l, high=x_u, size=(Bdry_collocation,1))
    y_Bdry = np.random.uniform(low=y_l, high=y_u, size=(Bdry_collocation,1))       
    
    #Move to pytorch tensors
    x_Bdry= Variable(torch.from_numpy(x_Bdry).float(), requires_grad=True).to(device)
    y_Bdry = Variable(torch.from_numpy(y_Bdry).float(), requires_grad=True).to(device)
    
    ##Pick pts to make tensor for No-Slip Boundary Condition
    x_l_Bdry = Variable(x_l * torch.ones_like(x_Bdry), requires_grad=True).to(device)
    x_u_Bdry = Variable(x_u * torch.ones_like(x_Bdry), requires_grad=True).to(device)
    y_l_Bdry = Variable(y_l * torch.ones_like(x_Bdry), requires_grad=True).to(device)
    y_u_Bdry = Variable(y_u * torch.ones_like(x_Bdry), requires_grad=True).to(device)
    
            
    return x_domain, y_domain, t_zero, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry
    
def tsliceTrainPts(net, Domain_collocation, Bdry_collocation, final_time):
    #Set of all the recorded t variable as base data for chasing during training

    #time starts at lower bound 0, ends at upper bouund updated in slices
    t_l = net.t_l

    #Pick IC/NSpde Condition Training Random Points in Numpy
    t_domain = np.random.uniform(low=t_l, high=final_time, size=(Domain_collocation, 1))
    
    #Move to pytorch tensors
    t_domain = Variable(torch.from_numpy(t_domain).float(), requires_grad=True).to(device)

    #Pick IC Training t starting points to make tensor
    t_zero = Variable(torch.zeros_like(t_domain), requires_grad=True).to(device)

    #Pick BC Training Random Points in Numpy
    t_Bdry = np.random.uniform(low=t_l, high=final_time, size=(Bdry_collocation,1))
    
    #Move to pytorch tensors
    t_Bdry = Variable(torch.from_numpy(t_Bdry).float(), requires_grad=True).to(device)
        
    return t_domain, t_Bdry
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def Create_IC_Parameters(x_domain, y_domain, t_zero, iterations, learning_rate, filename, record_loss, print_loss):
    ICnet = Net().to(device)
    
    IC_Only_training(ICnet, x_domain, y_domain, t_zero, iterations, learning_rate, record_loss, print_loss)
    
    torch.save(ICnet.state_dict(), filename)
    

def IC_Only_training(net, x_domain, y_domain, t_zero, iterations, learning_rate, record_loss, print_loss):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #learning rate update
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    #training loop
    epsilon_IC = [] #placeholder to track decreasing loss
    for epoch in range(1, iterations+1):
        
    
        # Resetting gradients to zero
        net.optimizer.zero_grad()
           
        #Loss based on Initial Condition
        loss = lossIC(net, x_domain, y_domain, t_zero)
           
        loss.backward()
        
        # Gradient Norm Clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=2, error_if_nonfinite=True)

        #Gradient Value Clipping
        #nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        
        net.optimizer.step()
           
        #Print Loss every 1000 Epochs
        with torch.autograd.no_grad():
            
            if epoch%record_loss == 0:
                epsilon_IC = np.append(epsilon_IC, loss.cpu().detach().numpy())
            if epoch%print_loss == 0:
                print("Iteration:", epoch, "Initial Condition Loss:", loss.data)
    
    np.savetxt('epsilon_IC.txt', epsilon_IC)


def training_loop(net, x_domain, y_domain, t_zero, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry, time_slices, iterations, learning_rate, IC_coefficient, record_loss, print_loss):
    global epsilon
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #learning rate update
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    for final_time in time_slices:
        
        with torch.autograd.no_grad():
            print("Current Final Time:", final_time, "Current Learning Rate: ", get_lr(net.optimizer))  
        
        indicator = False
        reset_regularization = 1000
        
        #Iterate over these points
        
        t_domain, t_Bdry = tsliceTrainPts(net, Domain_collocation = int(1000), Bdry_collocation = int(100), final_time = final_time)    
        for epoch in range(1, iterations+1):
            # Loss calculation based on partial differential equation (PDE) 
            
            if epoch%reset_regularization == 0:
                indicator = False
    
            if epoch%reset_regularization != 0: #To detect error on forward/Backward, add hashtag on this whole line, and
            #with torch.autograd.detect_anomaly(): #use this line alternatively by deleting hashtag.
                
                ###Training steps
                # Resetting gradients to zero
                net.optimizer.zero_grad()
            
                #Loss based on Initial Condition
                mse_IC = lossIC(net, x_domain, y_domain, t_zero)
                # Gradient Norm Clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

                #Loss based on Boundary Condition (Containing No-Slip and Free-slip)
                mse_BC = lossBdry(net, x_Bdry, y_Bdry, t_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry)
                # Gradient Norm Clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

                #Loss based on PDE
                mse_NS = lossNSpde(net, x_domain, y_domain, t_domain)
                # Gradient Norm Clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            
                if indicator == False:
                    indicator = True
                    IC_regular = mse_IC.detach()
                    BC_regular = mse_BC.detach()
                    pde_regular = mse_NS.detach()
                
                raw_loss = IC_coefficient * mse_IC + mse_BC + mse_NS
            
                mse_IC = mse_IC #/IC_regular
                mse_BC = mse_BC #/BC_regular
                mse_NS = mse_NS #/pde_regular
            
                #Combine all Loss functions
                loss = mse_BC + 10**5 *mse_IC  + 10**5 * mse_NS #IC_coefficient *
                # Gradient Norm Clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

                loss.backward()
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            #Gradient Value Clipping
            #nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            net.optimizer.step()
            
            #Print Loss every 1000 Epochs
            with torch.autograd.no_grad():
                if epoch%record_loss == 0:
                    epsilon = np.append(epsilon, raw_loss.cpu().detach().numpy())
                if epoch%print_loss == 0:
                    print("Iteration:", epoch, "\tTotal Loss:", loss.data)
                    print("IC Loss: ", mse_IC.data, "\tBC Loss: ", mse_BC.data, "\tNS PDE Loss: ", mse_NS.data)

            
                
create_network(True)
create_network(False)


# In[ ]:





# In[ ]:




