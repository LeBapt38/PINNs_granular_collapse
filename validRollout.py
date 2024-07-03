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
from matplotlib import pyplot as plt
import matplotlib.animation as animation

N_train = 5000
nbTrainingSteps = 0
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

#%% Préparation des données
t_star = np.linspace(0, 10, 100)
X_star = np.linspace(0,1, 100)
Y_star = np.linspace(0,0.2, 20)

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

# Rearrange Data
XX = np.tile(X, (1, T))  # N x T
YY = np.tile(Y, (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T



x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

dataset = TensorDataset(x, y, t)
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
print('Data ready')
#%% Validation du réseau
#Evaluate the neural network on the training data
pinn = PINNs.NavierStokes(dataloader)
pinn.net.load_state_dict(torch.load('model' + str(nbTrainingSteps) + '.pt'))
pinn.net.eval()

#Prepare the data for the rollout
x_test = X
y_test = Y
t_test = np.ones(len(x_test)) 

x_test = x_test.flatten()[:, None]  # NT x 1
y_test = y_test.flatten()[:, None]  # NT x 1
t_test = t_test.flatten()[:, None]  # NT x 1

x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)
x_test = x_test.to(pinn.device)
y_test = y_test.to(pinn.device)
t_test = t_test.to(pinn.device)

#%% The rollout with the animations
#Do the rollout
u_out, v_out, p_out, phi_out, f_out, g_out, h_out = pinn.function(x_test, y_test, t_test)

# Do the animation
u_plot = phi_out.data.cpu().numpy()
u_plot = np.reshape(u_plot, (20, 100))

fig, ax = plt.subplots()

plt.contourf(u_plot, levels=30, cmap='jet')
plt.colorbar()

def animate(i):
    ax.clear()
    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i*t_test)
    u_plot = phi_out.data.cpu().numpy()
    u_plot = np.reshape(u_plot, (100, 20))
    cax = ax.contourf(u_plot, levels=20, cmap='jet')
    plt.xlabel(r'$x$')
    plt.xlabel(r'$y$')
    plt.title(r'$p(x,\; y, \; t)$')

# Call animate method
#ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
#ani.save('p_field_lbfgs.gif')
#plt.close()
# Display the plot
plt.show()

















