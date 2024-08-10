# Physics Informed Neural Network(PINN) of the Rising Bubble Dynamics


This folder is code for our paper simulating the rising bubble dynamics by using PINN. There are two versions of governing equation of the rising bubble, in sharp interface ver. and diffuse interface ver. for the boundary of bubble.

You can access to the specific sharp/diffuse interface version project via: [[Sharp Interface]](https://github.com/woooojng/Bubble_PINN/tree/main/Sharp_interface_Bubble_PINN_ver1#physics-informed-neural-networkpinn-of-the-rising-bubble-dynamics) [[Diffuse Interface]](https://github.com/woooojng/Bubble_PINN/tree/main/Diffuse_interface_Bubble_PINN_ver1#physics-informed-neural-networkpinn-of-the-rising-bubble-dynamics).

[comment]: # ([[ResearchGate]])

## Problem Summary

This project presents methods to improve the performance of Physics Informed Neural Networks (PINNs). The enhanced PINNs are used to address complex problems involving rising bubble systems with free (moving) boundaries over time which is known as the most difficult part to simulate rising bubble. By integrating a time-adaptive approach with the level set method, the study aims to accurately model the dynamics of these moving boundaries in bubble systems.



## Abstract

The Physics-Informed Neural Network (PINN) merges neural networks (NN) with partial differential equations (PDEs), enabling direct solutions to intricate physical problems without strict dependence on labeled data. This cutting-edge approach synthesizes PDE principles with NN architecture to accurately predict system behavior, proving invaluable across diverse scientific and engineering domains.

This project introduces strategies to enhance PINN approximating capabilities. Specifically, the study focuses on applying these enhanced PINNs to solve complex problems related to rising bubble systems with free boundaries(moving boundaries as time goes by), employing a time-adaptive approach in conjunction with the level set method. By simulating using the PINN framework and comparing outcomes with existing results, the research aims to assess qualitative patterns and identify potential novel insights. Furthermore, utilizing existing data to validate accuracy and refine the network through the PINN procedure may reveal convergence among different methods and shed light on the true behavior of the system. Additionally, exploring the Deep Ritz method, which shares similarities with PINNs, could provide deeper insights into the underlying energy minimization associated with the problem when compared against PINN outcomes.

## Difficulties on each step to develop PINN model
Error on Gradient Blowing

Fine Tuning - Transfer learning on pre-training

Making bubble shape as time goes by further from the given initial bubble shape - Transfer learning by combining 2nd network
## Contact

wki1 [AT] iu [DOT] edu

[comment]: # (## License)

[comment]: # (MIT)
