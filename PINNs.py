#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:01:32 2024

@author: Baptiste Guilleminot
"""

import torch
import torch.nn as nn
import numpy as np
from math import exp

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

## Definis les conditions intitiales
def ci(x,y) :
    if x > 0.205 or y > 0.105 :
        return 0
    phi = 1
    if x < 0.205 and x > 0.195 :
        x0 = x - 0.195
        phi = exp(0.01005**2 / (x0**2 - 0.01005**2)) / exp(-1)
    if y < 0.105 and y > 0.095 :
        y0 = y - 0.095
        phi = exp(0.01005**2 / (y0**2 - 0.01005**2)) / exp(-1)
    return phi

## Definis la classe pour le PINN
class NavierStokes():
    def __init__(self, X, Y, T):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.t = self.t.to(self.device)

        #null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1)).to(self.device)
        # initialize network:
        self.network()
        self.net = self.net.to(self.device)

        self.LBFGS_optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                                 lr=0.5, 
                                                 max_iter=100, max_eval=None, 
                                                 history_size=50, 
                                                 tolerance_grad=1e-06, 
                                                 tolerance_change=1 * np.finfo(float).eps,
                                                 line_search_fn="strong_wolfe")
        
        self.adam_optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.mse = nn.MSELoss()

        #loss
        self.ls = 0

        #iteration number
        self.iter = 0

    def network(self):

        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 3))

    def function(self, x, y, t):
        
        #Get the results of the NN
        res = self.net(torch.hstack((x, y, t)))
        psi, p, phi = res[:, 0:1], res[:, 1:2], res[:, 2:3]
        
        #Modify and get related results
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0] #retain_graph=True,
        v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_t = torch.autograd.grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        
        
        eps_dot = torch.sqrt_(u_x**2 + v_y**2 + 0.5 * (v_x + u_y)**2)
        e = (mu_s * p) / (eps_dot + lamb) + (delta_mu * p) / (I0 * torch.sqrt_(torch.abs_(phi * p)) + eps_dot + lamb)
        e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
        e_y = torch.autograd.grad(e, y, grad_outputs=torch.ones_like(e), create_graph=True)[0]
        
        sigxx_x = e_x * u_x + e * u_xx
        sigyy_y = e_y * v_y + e * v_yy
        sigxy_y = 0.5 * e_y * (u_y + v_x) + 0.5 * e * (u_yy + v_xy)
        sigyx_x = 0.5 * e_x * (u_y + v_x) + 0.5 * e * (u_xy + v_xx)
        
        f = phi_t + u * phi_x + phi * u_x + v * phi_y + phi * v_y
        g = u_t + u * u_x + v * u_y + p_x - sigxx_x - sigxy_y
        h = v_t + u * v_x + v * v_y + p_y + rho * phi * G - sigyx_x - sigyy_y

        return u, v, p, phi, f, g, h
    
    def loss_function(self) :
        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, phi_prediction, f_prediction, g_prediction, h_prediction = self.function(self.x, self.y, self.t)

        # calculate losses
        u_loss = 0
        v_loss = 0
        phi_loss = 0
        p_loss = 0
        for i in range(self.x.shape[0]) :
            n = 0
            if self.x[i,0] == 0. or self.x[i,0] == L or self.y[i,0] == 0. or self.y[i,0] == h :
                u_loss += u_prediction[i,0]**2
                v_loss += v_prediction[i,0]**2
                n += 1
            u_loss /= (n+1)
            k = 0
            if self.t[i,0] == 0 :
                k += 1
                phi_loss += (phi_prediction[i,0] - ci(self.x[i, 0], self.y[i,0]))**2
            phi_loss /= (k+1)
            if self.x[i,0] > 0.75 and self.y[i,0] > 0.15 : 
                p_loss = (p_prediction[-1] - p0)**2
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        h_loss = self.mse(h_prediction, self.null)
        self.ls = u_loss + v_loss + p_loss + phi_loss + f_loss +g_loss + h_loss

        # derivative with respect to net s weights:
        self.ls.backward()
        
    def closure(self):
        # reset gradients to zero:
        self.LBFGS_optimizer.zero_grad()

        self.loss_function()

        self.iter += 1
        if not self.iter % 1:
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls.item()))

        return self.ls

    def LBFGS_train(self):

        # training loop
        self.net.train()
        self.LBFGS_optimizer.step(self.closure)
    
    def Adam_train(self, nb_epoch) :
        for i in range(nb_epoch) :
            self.adam_optimizer.zero_grad()
            self.loss_function()
            self.adam_optimizer.step()
            print(f'Adam iteration {i}, Loss: {self.ls.item():.6f}')
            
    def train(self, nb_epochs_adam, nb_epoch_lbfgs) :
        self.Adam_train(nb_epochs_adam)
#        for i in range(nb_epoch_lbfgs) :
#            self.LBFGS_train()
        










