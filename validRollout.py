#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:46:23 2024

@author: Baptiste Guilleminot
"""

import PINNs
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

N_train = 5000
nbTrainingSteps = 100
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
t_star = np.linspace(0, 10, 1000)
X_star = [np.linspace(0,0.2, 400), np.linspace(0,1, 2000)]

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T


x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]

#%% Validation du réseau
#Evaluate the neural network on the training data
pinn = PINNs.NavierStokes(x_train, y_train, t_train)
pinn.net.load_state_dict(torch.load('model' + str(nbTrainingSteps) + '.pt'))
pinn.net.eval()

#Prepare the data for the rollout
x_test = X_star[:, 0:1]
y_test = X_star[:, 1:2]
t_test = np.ones((x_test.shape[0], x_test.shape[1]))

x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)

#%% The rollout with the animations
#Do the rollout
u_out, v_out, p_out, phi_out, f_out, g_out, h_out = pinn.function(x_test, y_test, t_test)

# Do the animation
u_plot = phi_out.data.cpu().numpy()
u_plot = np.reshape(u_plot, (50, 100))

fig, ax = plt.subplots()

plt.contourf(u_plot, levels=30, cmap='jet')
plt.colorbar()

def animate(i):
    ax.clear()
    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i*t_test)
    u_plot = phi_out.data.cpu().numpy()
    u_plot = np.reshape(u_plot, (50, 100))
    cax = ax.contourf(u_plot, levels=20, cmap='jet')
    plt.xlabel(r'$x$')
    plt.xlabel(r'$y$')
    plt.title(r'$p(x,\; y, \; t)$')

# Call animate method
ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
#ani.save('p_field_lbfgs.gif')
#plt.close()
# Display the plot
plt.show()

















