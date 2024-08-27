# Physics Informed Neural Network(PINN) of the Rising Bubble Dynamics


This folder is code for our paper simulating the rising bubble dynamics by using PINN. Governing equation of the rising bubble is represented in the paper [1][HKBGT07] in the reference below.

You can download the paper via: [[ResearchGate]](https://www.researchgate.net/publication/228949659_Proposal_for_quantitative_benchmark_computations_of_bubble_dynamics).


[comment]: # ([[ResearchGate]])

## Problem Summary

This project presents methods to improve the performance of Physics Informed Neural Networks (PINNs). The enhanced PINNs are used to address complex problems involving rising bubble systems with diffuse boundaries over time which is known as the most difficult part to simulate rising bubble. By integrating a time-adaptive approach with the level set method, the study aims to accurately model the dynamics of these diffuse boundaries in bubble systems.


## Abstract

The Physics-Informed Neural Network (PINN) merges neural networks (NN) with partial differential equations (PDEs), enabling direct solutions to intricate physical problems without strict dependence on labeled data. This cutting-edge approach synthesizes PDE principles with NN architecture to accurately predict system behavior, proving invaluable across diverse scientific and engineering domains.

This project introduces strategies to enhance PINN approximating capabilities. Specifically, the study focuses on applying these enhanced PINNs to solve complex problems related to rising bubble systems with diffuse boundaries  as time goes by, employing a time-adaptive approach in conjunction with the level set method. By simulating using the PINN framework and comparing outcomes with existing results, the research aims to assess qualitative patterns and identify potential novel insights. Furthermore, utilizing existing data to validate accuracy and refine the network through the PINN procedure may reveal convergence among different methods and shed light on the true behavior of the system. Additionally, exploring the Deep Ritz method, which shares similarities with PINNs, could provide deeper insights into the underlying energy minimization associated with the problem when compared against PINN outcomes.

In this notebook, we are going to combine different two networks to apply the adaptive time marching strategy in the paper [1]. To describe in detail for this strategy, by discretizing time domain evenly, one network simulate PDE system on a time period and the next similar but different network simulate successively from the ending point of the former network. Therefore, as the name of itself, this strategy is applying simulation adaptively by marching on discretized time domain.

On our simulation, the first network on time $\[0, .1\]$ is made with the loss function added up for the loss terms coming from initial configuration equations, boundary condition equations and the following NS PDE equations.(See [1])


$$
    \rho (x) \left( \frac{\partial u}{\partial t} + u \cdot \nabla u \right) = - \nabla p +  \eta({\phi }) \Delta u + \rho (x) g,
$$
$$
    \nabla \cdot u = 0,$$
$$
    [u]|_\Gamma = 0, $$
$$
    [pI + \eta (\nabla u + (\nabla u)^T)]|_\Gamma \cdot n = \sigma_{DA} \kappa n .
$$

## Requirement

- Python 3.6
- PyTorch 2.3.0
- NumPy 1.22.4
- ‎Matplotlib 3.5.3 

## Preparation

### Clone

```bash
git clone https://github.com/hiyouga/RepWalk.git](https://github.com/woooojng/Bubble_PINN.git
```

[comment]: # (%### Create an anaconda environment [Optional]:)


[comment]: # (### Download the pretrained embeddings:)


## Usage

### Train the model at clonned `Diffuse_interface_Bubble_PINN_ver1` directory in terminal:

```bash
python3 train.py
```

### Show help message and exit:

```bash
python3 train.py -h
```

## File Specifications

- **Forward_with_Layer_Setting.py**: Neural Network Architecture for layer setting and forward step with x, y, t input/ velocity u = (u1, u2), pressure P, representation phi(yilds density rho later) and chemical potential m_D output variables.
- 
- **InitialConditionLoss.py**: For the equations ?? in chapter ?? in [2], Binary Cross Entropy loss function and MSE function associated with these equations is defined.
- **BoundaryLoss.py**: For the equations on left/right wall and top/bottom outer boundary of domain in chapter 2.2 in [1], MSE function associated with these equations is defined.
- **MvBdry_Coefficient_and_Loss.py**: For the equations on moving boundary sharp interface in chapter 2.1 in [1], MSE loss function associated with these equations is defined.
- **NSpde_loss.py**: For the Navier-Stokes PDE equations in chapter 2.1 in [1], MSE loss function associated with these equations is defined.
- **Train.py**: Neural network training function starting from Initial condition train running and then running for total loss summing with all loss functions.
- **NNlayers_Bubble_0.pt**: Result record of total loss training for network parameter with function `net.state_dict()`, based on summing up all the loss functions above.
- **IC_Only.pt**: Result record of Initial condition Binary cross entropy and MSE loss training for network parameter with function `net.state_dict()`, based only on initial condition loss function above to set initial condition firstly before total training. This training process is important to initialize initial moving boundry of the bubble.
- **visualize_Bubble_PINN.py**: Result graph of first initial condition training before total training. For initial configuration at time 0, we was able to see the t=0 starting circlular bubble with this graph.
- **Initial_Condition_Result_Circle.pdf**: The pdf file of the graph of 'visualize_Bubble_PINN.py'.
- **Train_whole_loop_6.22.html**: Result of whole Train process, containing Initial condition 30,000 iterations training with loss 0.0005 and total loss of all conditions 160,000 iterations training with learning rate .0001, .00001, .000005, .000001.

## Reference

[comment]: # (If this work is helpful, please cite as:)

<a id="1">[1]</a> 
S. Hysing,
S. Turek,
D. Kuzmin,
N. Parolini, E. Burman,
S. Ganesank, and L. Tobiska, 
Proposal for quantitative benchmark
computations of bubble dynamics, 
Ergebnisberichte des Instituts für Angewandte Mathematik, Nummer
351, Fakultät für Mathematik, TU Dortmund, 2007.


<a id="1">[2]</a> 
T.H.B. Demonta, S.K.F. Stotera, E.H. van Brummelen, 
Numerical Investigation of the Sharp-Interface Limit of the
Navier-Stokes-Cahn-Hilliard Equations, Journal of Fluid Mechanics,  970 , art. no. A24, 2023.

[comment]: # (## Acknowledgments)

[comment]: # (This work is supported partly by the National Natural Science Foundation)

## Contact

wki1 [AT] iu [DOT] edu

[comment]: # (## License)

[comment]: # (MIT)
