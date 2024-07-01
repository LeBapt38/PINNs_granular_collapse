#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:35:33 2024

@author: Baptiste Guilleminot
"""
import PINNs
import torch
import numpy as np

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
t_star = np.linspace(0, 10, 1000)
X_star = np.linspace(0,1, 1000)
Y_star = np.linspace(0,0.2, 200)

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


#%% Entrainement
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
pinn = PINNs.NavierStokes(x_train, y_train, t_train)
pinn.train(1000,5)
torch.save(pinn.net.state_dict(), 'model0.pt')

for i in range(nbTrainingSteps) :
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    pinn = PINNs.NavierStokes(x_train, y_train, t_train)
    pinn.net.load_state_dict(torch.load('model' + str(i) +'.pt'))
    pinn.train(1000,5)
    torch.save(pinn.net.state_dict(), 'model' + str(i+1) + '.pt')
















