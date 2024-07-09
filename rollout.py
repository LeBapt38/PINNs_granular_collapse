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
from matplotlib.ticker import MaxNLocator

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

#%% Prepare the data
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
dataloader_test = DataLoader(dataset, batch_size=N, shuffle=False)
print('Input ready')

#%% Create the output from desired data
pinn = PINNs.NavierStokes(dataloader_test)
pinn.net.load_state_dict(torch.load(model))
data = pinn.rollout(len(X_star), len(Y_star), value)
data = np.array(data)
maxi = np.max(data)
mini = np.min(data)
print('rollout done')
#%% Create the animation
x_labels = [f"{label:.2f}" for label in X_star]
y_labels = [f"{label:.2f}" for label in Y_star]
fig, ax = plt.subplots()
fig.suptitle('Rollout of PINNs')
cax = ax.matshow(data[0], cmap='jet', origin='lower', vmin = mini, vmax = maxi)
cbar = fig.colorbar(cax)
cbar.set_ticks(np.linspace(mini, maxi, 5))
cbar.set_label('Values of ' + value)
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('outward', 10))
ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax.set_aspect(3)
def update(frame):
    cax.set_data(data[frame])
    return [cax]
interval = 1000 * (t_star[-1] - t_star[0]) / len(t_star)
interval = int(interval)
ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
if not save :
    plt.show()
else : 
    ani.save('grid_animation.mp4', writer='ffmpeg')