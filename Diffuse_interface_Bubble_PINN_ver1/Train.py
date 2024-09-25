#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
import time
import shutil
import os
from datetime import datetime
currentDateTime = datetime.now() 
print("Date of Today : ", currentDateTime.month, " /", currentDateTime.day, "\nHour : ", currentDateTime.hour) 

#Date of Today
ctime = f"{currentDateTime.month}_{currentDateTime.day}_{currentDateTime.hour}h"


#Call model of layers and its forward step
from Forward_with_Layer_Setting import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Call training functions of Loss functions
from NSpde_loss import lossNSpde
from BoundaryLoss import lossBdry
from InitialConditionLoss import lossIC
from secondInitialConditionLoss import second_net_lossIC
from Dummy_Layer_Setting import Dummy_Net

def create_network(fine_tuning, next_network):
    
    net = Net()
    net = net.to(device)

    #Set final times for running training
    time_slices = np.array([.1, .3, .6, 1]) #, .2, .4, .6, .8, 1
    
    start = time.time()

    #Start transfer the pre-training stat_dic parameters
    if fine_tuning == True:
        #time_slices = np.array([1])
        #copy the source file
        print('Transfer Learning ; Training with Pre-training Parameters')
        shutil.copy('**.pt', '**_Transfer.pt')
        net.load_state_dict(torch.load("**_Transfer.pt"))
        print("  - Pre-training Loaded\n")

        
    time_vec = [0, 0, 0, 0, 0]
    
    #attempt to load IC if it exists
    #try:
        #net.load_state_dict(torch.load("IC_Only.pt"))
        #net.load_state_dict(torch.load("NNlayer_Bubble_4.pt"))
        #print("Previous State Loaded")
    #except:
        #pass
    
    global epsilon #used to track loss
    epsilon = []
    
    print('=======================================================Training PDE Start=======================================================')
    
    for i in range(5):
        #Set loop to optimize in progressively smaller learning rates
        if i == 0:
            #First loop uses progressively increasing time intervals
            print('\n\nExecuting Pass 1')
            iterations = 100000
            learning_rate = 5*10**-3    
        elif i == 1:
            print('\n\nExecuting Pass 2')
            time_slices = [time_slices[-1]]
            iterations = 60000 #50000
            learning_rate = 5*10**-4
        elif i == 2:
            print('\n\nExecuting Pass 3')
            iterations = 60000
            learning_rate = 5*10**-5
        elif i ==3:
            print('\n\nExecuting Pass 4')
            iterations = 60000
            learning_rate = 5*10**-6
        elif i ==4:
            print('\n\nExecuting Pass 5')
            iterations = 25000
            learning_rate = 5*10**-7
        
        training_loop(net, time_slices, iterations, learning_rate, IC_coefficient = 1, record_loss = 100, print_loss = 500)
        torch.save(net.state_dict(), f"{ctime}_NNlayers_Bubble_{i}.pt")
        np.savetxt('epsilon.txt', epsilon)
        time_vec[i] = time.time()

    np.savetxt('epsilon.txt', epsilon)
    
    end = time.time()

    print("Total Time:\t", end-start, '\nPass 1 Time:\t', time_vec[0]-start, '\nPass 2 Time:\t', time_vec[1]-start, '\nPass 3 Time:\t', time_vec[2]-start, '\nPass 4 Time:\t', time_vec[3]-start, '\nPass 5 Time:\t', time_vec[4]-start)
    
    #Start 2nd Network with the result of 1st network in the place of IC
    if next_network == True:
        print('\n\nStarting 2nd Network by Trasfering 1st Network')
        
        starting_time = 0.1
        time_slices = np.array([.2]) #.3, .4, .5, .6, .7, .8, .9, 1
        
        #copy the layer parameter file we target as previous layer
        ###Change this .pt file to continue the network we want
        
        shutil.copy('ES_min_loss_lr1e-07_t0.1_8_16_17h.pt', 'ES_min_loss_lr1e-07_t0.1_8_16_17h_prep2nd.pt')
        
        #load the layer parameter file
        net.load_state_dict(torch.load("ES_min_loss_lr1e-07_t0.1_8_16_17h_prep2nd.pt"))
        load_IC = "ES_min_loss_lr1e-07_t0.1_8_16_17h_prep2nd.pt"
        #Prep variable for getting the final output of 1st Net. to take as target value of IC loss
        
        
        print(f"  - 1st Network Layers '{load_IC}' Loaded\n\n")
        
        
        time_vec = [0, 0, 0, 0, 0]
    
    
        print('=======================================================2nd Training Start=======================================================')
    
        for i in range(5):
            #Set loop to optimize in progressively smaller learning rates
            if i == 0:
                #First loop uses progressively increasing time intervals
                print('\n\nExecuting Pass 1')
                iterations = 90000 #100000
                learning_rate = 10**-3  
            elif i == 1:
                print('\n\nExecuting Pass 2')
                time_slices = [time_slices[-1]]
                iterations = 60000 #50000
                learning_rate = 10**-4
            elif i == 2:
                print('\n\nExecuting Pass 3')
                iterations = 60000
                learning_rate = 10**-5
            elif i ==3:
                print('\n\nExecuting Pass 4')
                iterations = 60000
                learning_rate = 10**-6
            elif i ==4:
                print('\n\nExecuting Pass 5')
                iterations = 60000
                learning_rate = 10**-7
        
            #Transfer learning as new IC loss term
            next_training_loop(net, time_slices, iterations, learning_rate, IC_coefficient = 1, record_loss = 100, print_loss = 500, starting_time = starting_time, load_IC = load_IC)
        
            torch.save(net.state_dict(), f"{ctime}__Net2_NNlayers_Bubble_{i}.pt")
            np.savetxt('epsilon.txt', epsilon)
            time_vec[i] = time.time()

        np.savetxt('epsilon.txt', epsilon)
    
        #Record time
        second_net_done = time.time()
        
        print('Total Time until 2nd Net:\t', second_net_done-start)
        
        

def twoDimTrainPts(net, Domain_collocation, Bdry_collocation, starting_time):
    #Set of all the recorded xy variables as base data for chasing during training
    
    # Domain boundary in the range [0, 1]x[0, 2] and time in [0, 1].
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u

    #time starts at lower bound 0, ends at upper bouund updated in slices
    t_l = starting_time

    #Pick IC/NSpde Condition Training Random Points in Numpy
    x_domain = np.random.uniform(low= x_l, high=x_u, size=(Domain_collocation, 1)) 
    y_domain = np.random.uniform(low= y_l, high=y_u, size=(Domain_collocation, 1)) 
    
    #Move to pytorch tensors
    x_domain = Variable(torch.from_numpy(x_domain).float(), requires_grad=True).to(device)
    y_domain = Variable(torch.from_numpy(y_domain).float(), requires_grad=True).to(device)
    
    #Pick IC Training t starting points to make tensor
    t_lower = Variable(t_l * torch.ones_like(x_domain), requires_grad=True).to(device)

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
    
            
    return x_domain, y_domain, t_lower, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry
    
def tsliceTrainPts(net, Domain_collocation, Bdry_collocation, starting_time, final_time):
    #Set of all the recorded t variable as base data for chasing during training

    #time starts at lower bound 0, ends at upper bouund updated in slices
    t_l = starting_time

    #Pick IC/NSpde Condition Training Random Points in Numpy
    t_domain = np.random.uniform(low=t_l, high=final_time, size=(Domain_collocation, 1))
    
    #Move to pytorch tensors
    t_domain = Variable(torch.from_numpy(t_domain).float(), requires_grad=True).to(device)

    #Pick IC Training t starting points to make tensor
    #t_zero = Variable(t_l *torch.ones_like(t_domain), requires_grad=True).to(device)

    #Pick BC Training Random Points in Numpy
    t_Bdry = np.random.uniform(low=t_l, high=final_time, size=(Bdry_collocation,1))
    
    #Move to pytorch tensors
    t_Bdry = Variable(torch.from_numpy(t_Bdry).float(), requires_grad=True).to(device)
        
    return t_domain, t_Bdry
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def next_training_loop(net, time_slices, iterations, learning_rate, IC_coefficient, record_loss, print_loss, starting_time, load_IC):
    global epsilon
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_l = starting_time

    
    #learning rate update
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    for final_time in time_slices:
        min_loss = 15
        
        with torch.autograd.no_grad():
            print("\n\nCurrent Starting Time:", t_l, "  Current Final Time:", final_time, "\nCurrent Learning Rate: ", get_lr(net.optimizer))  
        
        #Iterate over these points
            
        for epoch in range(1, iterations+1):
            # Loss calculation based on partial differential equation (PDE) 

            #Sample Collocation Points
            t_domain, t_Bdry = tsliceTrainPts(net, Domain_collocation = int(20000), Bdry_collocation = int(5000), starting_time = t_l, final_time = final_time)
            x_domain, y_domain, t_l_domain, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry = twoDimTrainPts(net, Domain_collocation = int(20000), Bdry_collocation = int(5000), starting_time = t_l)
            x_IC, y_IC, t_l_IC, _, _, _,  _,_,_ = twoDimTrainPts(net, Domain_collocation = int(10000), Bdry_collocation = int(1), starting_time = t_l)
            
                
            ###Training steps
            # Resetting gradients to zero
            net.optimizer.zero_grad()
            
            #Loss based on Initial Condition
            mse_IC = second_net_lossIC(net, x_IC, y_IC, t_l_IC, load_IC)
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            #Loss based on Boundary Condition (Containing No-Slip and Free-slip)
            mse_BC_u, mse_BC_phi, mse_BC_mu = lossBdry(net, x_Bdry, y_Bdry, t_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry)
            mse_BC = mse_BC_u + mse_BC_phi + mse_BC_mu
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            #Loss based on PDE
            mse_NS, mse_NSdiv, mse_CH, mse_CHdecoup = lossNSpde(net, x_domain, y_domain, t_domain)
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            if mse_BC_u > .01:
                #Combine all Loss functions
                loss =  100*mse_BC + 200 *mse_IC+ 2*(mse_NS + mse_NSdiv  + mse_CH + mse_CHdecoup)
            else:
                loss =  10*mse_BC + 2000 *mse_IC+ 2*(mse_NS + mse_NSdiv  + mse_CH + mse_CHdecoup)
            
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
                    epsilon = np.append(epsilon, loss.cpu().detach().numpy())
                if epoch%print_loss == 0:
                    print("Iteration:", epoch, "\tTotal Loss:", loss.data)
                    print("IC Loss: ", mse_IC.data, "\tu BC Loss: ", mse_BC_u.data,  "\tphi BC Loss: ", mse_BC_phi.data, "\tmu BC Loss: ", mse_BC_mu.data, "\tNS PDE Loss: ", mse_NS.data, "\tNS Div Free Loss: ", mse_NSdiv.data, "\tCH Loss: ", mse_CH.data, "\tCH mu Loss: ", mse_CHdecoup.data)
                    
                    if mse_NS.cpu().detach().numpy() < 10**(-1) and mse_NS.cpu().detach().numpy() >= 10**(-2):
                        torch.save(net.state_dict(), f"ES_NSloss2nd_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS PDE Loss of 2nd decimal place\n')
                    if mse_NS.cpu().detach().numpy() < 10**(-2) and mse_NS.cpu().detach().numpy() >= 10**(-3):
                        torch.save(net.state_dict(), f"ES_NSloss3rd_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS PDE Loss of 3rd decimal place\n')
                    if mse_NS.cpu().detach().numpy() < 10**(-3) and mse_NS.cpu().detach().numpy() >= 10**(-4):
                        torch.save(net.state_dict(), f"ES_NSloss4th_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS PDE Loss of 4th decimal place\n')
                    if mse_NS.cpu().detach().numpy() < 10**(-4) and mse_NS.cpu().detach().numpy() >= 10**(-5):
                        torch.save(net.state_dict(), f"ES_NSloss5th_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS PDE Loss of 5th decimal place\n')
                    if mse_NS.cpu().detach().numpy() < 10**(-5) and mse_NS.cpu().detach().numpy() >= 10**(-6):
                        torch.save(net.state_dict(), f"ES_NSloss6th_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS PDE Loss of 6th decimal place\n')  
                        
                    if mse_IC.cpu().detach().numpy() < 7* 10**(-3) and mse_IC.cpu().detach().numpy() >= 10**(-6):
                        torch.save(net.state_dict(), f"ES_ICloss3rd_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest IC Loss of 3rd decimal place\n')
                    
                    if mse_IC.cpu().detach().numpy() < 7* 10**(-3) and mse_NS.cpu().detach().numpy() < 10**(-1):
                        torch.save(net.state_dict(), f"ES_ICloss3rd_NSloss2nd_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest NS Loss of 2nd decimal place and IC Loss of 3rd decimal place\n')
                    if mse_IC.cpu().detach().numpy() < 5* 10**(-3) and mse_NS.cpu().detach().numpy() < 5 * 10**(-4):
                        torch.save(net.state_dict(), f"ES_ICloss3rd_NSloss4th_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the latest IC Loss of 3rd decimal place and NS Loss of 4th decimal place\n')
                    if loss.cpu().detach().numpy() < min_loss:
                        min_loss = loss.cpu().detach().numpy()
                        torch.save(net.state_dict(), f"ES_min_loss_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}_Net2.pt")
                        print('  *Saved ; Early Stopping for the Minimal Total Loss\n')
                            
                        
        
        torch.save(net.state_dict(), f"BubbleLayers_lr{get_lr(net.optimizer)}_finaltime{final_time}_Net2.pt")
  
    
    
def training_loop(net, time_slices, iterations, learning_rate, IC_coefficient, record_loss, print_loss):
    global epsilon
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_l = 0
    
    
    #learning rate update
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    for final_time in time_slices:
        min_loss = 20
        min_ICloss = .1
        min_NSloss = .1
        mse_CHloss = .1
        
        with torch.autograd.no_grad():
            print("\n\nCurrent Final Time:", final_time, "\nCurrent Learning Rate: ", get_lr(net.optimizer))  
        
        indicator = False
        #reset_regularization = 1000
        
        #Iterate over these points
        
            
        for epoch in range(1, iterations+1):
            # Loss calculation based on partial differential equation (PDE) 

            #Sample Collocation Points
            t_domain, t_Bdry = tsliceTrainPts(net, Domain_collocation = int(10000), Bdry_collocation = int(500), starting_time = t_l, final_time = final_time)
            x_domain, y_domain, t_l_domain, x_Bdry, y_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry = twoDimTrainPts(net, Domain_collocation = int(10000), Bdry_collocation = int(500), starting_time = t_l)
            x_IC, y_IC, t_l_IC, _, _, _,  _,_,_ = twoDimTrainPts(net, Domain_collocation = int(10000), Bdry_collocation = int(1), starting_time = t_l)
            
            #if epoch%reset_regularization == 0:
                #indicator = False
    
            #if epoch%reset_regularization != 0: #To detect error on forward/Backward, add hashtag on this whole line, and
            #with torch.autograd.detect_anomaly(): #use this line alternatively by deleting hashtag.
                
            ###Training steps
            # Resetting gradients to zero
            net.optimizer.zero_grad()
            
            #Loss based on Initial Condition
            mse_IC = lossIC(net, x_IC, y_IC, t_l_IC)
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            #Loss based on Boundary Condition (Containing No-Slip and Free-slip)
            mse_BC_u, mse_BC_phi, mse_BC_mu = lossBdry(net, x_Bdry, y_Bdry, t_Bdry, x_l_Bdry, x_u_Bdry, y_l_Bdry, y_u_Bdry)
            mse_BC = mse_BC_u + mse_BC_phi + mse_BC_mu
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)

            #Loss based on PDE
            mse_NS, mse_NSdiv, mse_CH, mse_CHdecoup, Reg = lossNSpde(net, x_domain, y_domain, t_domain)
            # Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm= 5*10**2, norm_type=1, error_if_nonfinite=False)
                
            
                #if indicator == False:
                    #indicator = True
                    #IC_regular = mse_IC.detach()
                    #BC_regular = mse_BC.detach()
                    #pde_regular = mse_NS.detach()
                    #pdediv_regular = mse_NSdiv.detach()
                
                #raw_loss = IC_coefficient * mse_IC + mse_BC + mse_NS + mse_NSdiv + mse_CH + mse_CHdecoup
            
                #mse_IC = mse_IC #/IC_regular
                #mse_BC = mse_BC #/BC_regular
                #mse_NS = mse_NS #/pde_regular
                #mse_NSdiv = mse_NSdiv #/pdediv_regular
            
            #Combine all Loss functions
            loss =  mse_BC + 1000 *mse_IC+ 2*(mse_NS + mse_NSdiv  + mse_CH + mse_CHdecoup) + .1* Reg
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
                    epsilon = np.append(epsilon, loss.cpu().detach().numpy())
                if epoch%print_loss == 0:
                    print("Iteration:", epoch, "\tTotal Loss:", loss.data)
                    print("IC Loss: ", mse_IC.data, "\tu BC Loss: ", mse_BC_u.data,  "\tphi BC Loss: ", mse_BC_phi.data, "\tmu BC Loss: ", mse_BC_mu.data, "\tNS PDE Loss: ", 
                          mse_NS.data, "\tNS Div Free Loss: ", mse_NSdiv.data, "\tCH Loss: ", mse_CH.data, "\tCH mu Loss: ", mse_CHdecoup.data)
                    
                    if mse_IC.cpu().detach().numpy() < min_ICloss:
                        min_ICloss = mse_IC.cpu().detach().numpy()
                        torch.save(net.state_dict(), f"ES_min_ICloss_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}.pt")
                        print('  *Saved ; Early Stopping for the Minimal IC Loss\n')
                    
                    if mse_NS.cpu().detach().numpy() < min_NSloss:
                        min_NSloss = mse_NS.cpu().detach().numpy()
                        torch.save(net.state_dict(), f"ES_min_NSloss_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}.pt")
                        print('  *Saved ; Early Stopping for the Minimal NS Loss\n')
                        
                    if mse_CH.cpu().detach().numpy() < min_CHloss:
                        min_CHloss = mse_CH.cpu().detach().numpy()
                        torch.save(net.state_dict(), f"ES_min_CHloss_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}.pt")
                        print('  *Saved ; Early Stopping for the Minimal CH Loss\n')
                    
                    if loss.cpu().detach().numpy() < min_loss:
                        min_loss = loss.cpu().detach().numpy()
                        torch.save(net.state_dict(), f"ES_min_loss_lr{get_lr(net.optimizer)}_t{final_time}_{ctime}.pt")
                        print('  *Saved ; Early Stopping for the Minimal Total Loss\n')
                            
                        
        
        torch.save(net.state_dict(), f"BubbleLayers_lr{get_lr(net.optimizer)}_finaltime{final_time}.pt")
                    
#Printing on Wide Screen                    
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>:root { --jp-notebook-max-width: 100% !important; }</style>"))            
            
create_network(fine_tuning = False, next_network = False)
#create_network(fine_tuning = True, next_network = False)
#create_network(fine_tuning = False, next_network = True)
#create_network(fine_tuning = True, next_network = True)   


# In[ ]:




