#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:46:23 2024

@author: Baptiste Guilleminot
"""

import PINNs
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

nu = 0.01
rho = 2500
G = 9.81
mu_s = 0.383
lamb = 1e-6
delta_mu = 0.26
I0 = 0.279
p0 = 1e5
L = 1
h = 0.2

#%% Example data
t_star = np.linspace(0, 10, 100)
X_star = np.linspace(0,1, 100)
Y_star = np.linspace(0,0.2, 20)
save=False
value = "g"
model = 'model3450.pt'

#%% Préparation des données
X = []
Y = []
for i in range(len(Y_star)) :
    for j in range(len(X_star)) :
        X.append(X_star[j])
        Y.append(Y_star[i])
X = np.array(X)
Y = np.array(Y)

N = X.shape[0]
T = t_star.shape[0]

XX = np.tile(X, (T, 1))  # N x T
YY = np.tile(Y, (T,1))  # N x T
TT = np.tile(t_star, (N,1)).T  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

dataset = TensorDataset(x, y, t)
dataloader_valid = DataLoader(dataset, batch_size=N, shuffle=False)
print('Input ready')

#%% Validation du réseau
#Evaluate the neural network on the training data
pinn = PINNs.NavierStokes(dataloader_valid)
pinn.net.load_state_dict(torch.load(model))
print(pinn.valid())



