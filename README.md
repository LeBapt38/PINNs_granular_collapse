# Physicaly informed neural network (PINN) for granular collapse

### Table of content

- [Introduction](#introduction)
- [The problem to solve](#the-problem-to-solve)
- [Architecture](#architecture)
- [How it is coded](#how-it-is-coded)
- [The optimizers](#the-optimizers)
- [Limitations](#limitations)

## Introduction

The objective of this little Neural Network was to test the ability of a more simple form of AI compare to the GNS. The idea is to use a PINN to create a highly non linear function function which will be train to simulate a granular collapse.

This came to mind at the end of my internship, while seeing the limitations of the GNS algorithem. I wanted to create my own architecture in the little time I had left. In order to do so, it was convenient to take a PINN because I won't have to interface it with another algorithm like the MPM. 

## The problem to solve

I consider a granular collapse modelled by a viscosity "mu of I". the amount of material is computed with the ratio of material against air (byphasic equations). 

The boundary conditions are fixed with a no slip condition. The initial conditions are a bloc of sand in a "Heaviside way" a bit smoothed by a bits of the usual test function (see distribution theory).

## Architecture
**The inputs** are the space and time coordinates (2 dimension of space are considered).

**The raw outputs** are a potential for the speed, the ratio between sand and air and the pressure. These outputs are then **processed** to get the different speeds and their derivatives, the pressure, the ratio of sand (phi) and the different equations.

There are 10 **hidden layers** of 20 neurons each. The different layers are fully connected and the **actvation function** is tanh. 

## How it is coded

The neural network is coded in a class. For the instanciation, you have to give the data as a form of a dataloader like the one built in training.py.

The function takes an st of input and rollout the neural network. The outputs of this method are the processed one.

The loss_function method does the rollout and compute the loss function. It also run the loss function backward to get the derivatives with respect to each nodes.

Then there is the LBFGS_train method which train the neural network using the LBFGS optimizer. Another method was created for the Adam optimizer.

The train method is the one to use when training the network. It first use the Adam training version and then switch to the LBFGS one using the best situation to continue the training.

## The optimizers

I don't use one but 2 optimizers : 
- The first one is the classic **Adam**. It has the advantage to be quite efficient and allow to make the learning process quicker at the begining
- The second one is the more refined **LBFGS**. It is supposed to be more precise and more stable which is necessary at the end of the learning process.

## The tuning 

Different parameters have to be tuned like the learning rate and the mass of each component of the loss. More info when I found something stable.

## Limitations

The design of the program imply some limitations. For example, the fact that the network learn the all history at once means that the modifications of the boundary condition or the initial condition will be very limited. For larger changes in this aspect, the model may have to be retrained.

