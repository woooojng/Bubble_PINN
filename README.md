# Physics Informed Neural Network(PINN) of the Rising Bubble Dynamics


This folder is code for our paper simulating the rising bubble dynamics by using PINN. Governing equation of the rising bubble is represented in the paper [1][HKBGT07] in the reference below.

You can download the paper via: [[ResearchGate]](https://www.researchgate.net/publication/228949659_Proposal_for_quantitative_benchmark_computations_of_bubble_dynamics).


[comment]: # ([[ResearchGate]])

## Problem Summary

This project presents methods to improve the performance of Physics Informed Neural Networks (PINNs). The enhanced PINNs are used to address complex problems involving rising bubble systems with free (moving) boundaries over time which is known as the most difficult part to simulate rising bubble. By integrating a time-adaptive approach with the level set method, the study aims to accurately model the dynamics of these moving boundaries in bubble systems.

![](assets/example.jpg)

## Abstract

The Physics-Informed Neural Network (PINN) merges neural networks (NN) with partial differential equations (PDEs), enabling direct solutions to intricate physical problems without strict dependence on labeled data. This cutting-edge approach synthesizes PDE principles with NN architecture to accurately predict system behavior, proving invaluable across diverse scientific and engineering domains.

This project introduces strategies to enhance PINN approximating capabilities. Specifically, the study focuses on applying these enhanced PINNs to solve complex problems related to rising bubble systems with free boundaries(moving boundaries as time goes by), employing a time-adaptive approach in conjunction with the level set method. By simulating using the PINN framework and comparing outcomes with existing results, the research aims to assess qualitative patterns and identify potential novel insights. Furthermore, utilizing existing data to validate accuracy and refine the network through the PINN procedure may reveal convergence among different methods and shed light on the true behavior of the system. Additionally, exploring the Deep Ritz method, which shares similarities with PINNs, could provide deeper insights into the underlying energy minimization associated with the problem when compared against PINN outcomes.

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

### Train the model:

```bash
python3 train.py
```

### Show help message and exit:

```bash
python3 train.py -h
```

## File Specifications

- **Forward_with_Layer_Setting.py**: Neural Network Architecture for layer setting and forward step with x, y, t input/ velocity u, pressure P, density rho viscoscity mu output variables.
- **InitialConditionLoss.py**: For the equations in chapter 2.2 in [1], Binary Cross Entropy loss function and MSE function associated with these equations is defined.
- **BoundaryLoss.py**: For the equations on left/right wall and top/bottom outer boundary of domain in chapter 2.2 in [1], MSE function associated with these equations is defined.
- **MvBdry_Coefficient_and_Loss.py**: For the equations on moving boundary sharp interface in chapter 2.1 in [1], MSE loss function associated with these equations is defined.
- **NSpde_loss.py**: For the Navier-Stokes PDE equations in chapter 2.1 in [1], MSE loss function associated with these equations is defined.
- **Train.py**: Neural network training function starting from Initial condition train running and then running for total loss summing with all loss functions.
- 
- **NNlayers_Bubble_0.pt**:
- **IC_Only.pt**:
- **visualize_Bubble_PINN.py**: Result record of stat.
- **Initial_Condition_Result_Circle.pdf**: The scripts for training and evaluating the models.
- **Train_whole_loop_6.22.html**: Result of whole Train process, containing Initial condition 30,000 iterations training with loss 0.0005 and total loss of all conditions 160,000 iterations training with learning rate .0001, .00001, .000005, .000001.

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{zheng2020replicate,
  title={Replicate, Walk, and Stop on Syntax: an Effective Neural Network Model for Aspect-Level Sentiment Classification},
  author={Yaowei, Zheng and Richong, Zhang and Samuel, Mensah and Yongyi, Mao},
  booktitle={{AAAI}},
  year={2020}
}
```

[comment]: # (## Acknowledgments)

[comment]: # (This work is supported partly by the National Natural Science Foundation)

## Contact

wki1 [AT] iu [DOT] edu

[comment]: # (## License)

[comment]: # (MIT)
