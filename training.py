#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:35:33 2024

@author: Baptiste Guilleminot
"""
import PINNs
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

N_train = 2000
nbTrainingSteps = 2
nu = 0.01
rho = 2500
G = 9.81
mu_s = 0.383
lamb = 1e-3
delta_mu = 0.26
I0 = 0.279
p0 = 1e5
L = 1
h = 0.2

#%%Préparation des données
t_star = np.linspace(0, 10, 100)
X_star = np.linspace(0,1, 100)
Y_star = np.linspace(0,0.2, 20)

X = []
Y = []
for i in range(len(X_star)) :
    for j in range(len(Y_star)) :
        X.append(X_star[i])
        Y.append(Y_star[j])
X = np.array(X)
Y = np.array(Y)

N = X.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X, (1, T))  # N x T
YY = np.tile(Y, (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T



x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

print(x.shape, y.shape, t.shape)
x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

dataset = TensorDataset(x, y, t)
dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)


#%% Entrainement
def trainPINN(dataloader, nbTrainingSteps_adam = 30, nbTrainingSteps_lbfgs = 150, fromJsonFile = None) :
    pinn = PINNs.NavierStokes(dataloader, fromJsonFile=fromJsonFile)
    pinn.train(nbTrainingSteps_adam, nbTrainingSteps_lbfgs)
    pinn.savePINN('autobackup' + str(pinn.iter) + '.json')
    return(pinn.iter)


iteration = trainPINN(dataloader)
trainPINN(dataloader, fromJsonFile='autobackup' + str(iteration) + '.json')















