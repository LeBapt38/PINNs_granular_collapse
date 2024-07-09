#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:01:32 2024

@author: Baptiste Guilleminot
"""

import torch
import torch.nn as nn
import numpy as np
from math import exp, isnan
import json

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
    return 0.6 * phi

## Definis la classe pour le PINN
class NavierStokes():
    def __init__(self, dataloader, fromJsonFile = None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.sizeBatch = dataloader.batch_size

        #null vector to test against f and g:
        self.null = torch.zeros((self.sizeBatch, 1)).to(self.device)
        # initialize network:
        self.network()
        if fromJsonFile is not None : 
            with open(fromJsonFile, 'r') as file :
                data = json.load(file)
            self.net.load_state_dict(torch.load(data['net']))
            self.ls = data['loss']
            self.iter = data['iter']
            self.ls_fct_steps = data['lossFctSteps']
        else :
            self.ls = 0
            self.iter = 0
            self.ls_fct_steps = []
        self.net = self.net.to(self.device)

        self.LBFGS_optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                                 lr=0.0001, 
                                                 max_iter=1000, max_eval=None, 
                                                 history_size=50, 
                                                 tolerance_grad=1e-08, 
                                                 tolerance_change=0.01 * np.finfo(float).eps,
                                                 line_search_fn="strong_wolfe")
        
        self.adam_optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0005)
        self.mse = nn.MSELoss()

        self.ls_average = 0

    def network(self):

        self.net = nn.Sequential(
            nn.Linear(3, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 4))

    def function(self, x, y, t):
        
        #Get the results of the NN
        res = self.net(torch.hstack((x, y, t)))
        u, v, p, phi = res[:, 0:1], res[:, 1:2], res[:, 2:3], res[:, 3:4]
        p *= 1e5
        if isnan(p[0].item()) :
            print('p')
        elif isnan(phi[0].item()) :
            print('phi')
        
        #Modify and get related results
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
        g = rho * phi * (u_t + u * u_x + v * u_y) + p_x - sigxx_x - sigxy_y
        h = rho * phi * (v_t + u * v_x + v * v_y) + p_y + rho * phi * G - sigyx_x - sigyy_y

        return u, v, p, phi, f, g, h
    
    def loss_function(self, x, y, t) :
        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, phi_prediction, f_prediction, g_prediction, h_prediction = self.function(x, y, t)

        # calculate losses
        u_loss = 0
        v_loss = 0
        phi_loss = 0
        p_loss = 0
        n, k = 0, 0
        for i in range(self.sizeBatch) :
            if x[i,0] < 0.0001 or x[i,0] > L - 0.0001 or y[i,0] < 0.0001 or y[i,0] > h - 0.0001 :
                u_loss += u_prediction[i,0]**2
                v_loss += v_prediction[i,0]**2
                n += 1
            if t[i,0] == 0 :
                k += 1
                phi_loss += (phi_prediction[i,0] - ci(x[i, 0], y[i,0]))**2
            if y[i,0] > 0.15 : 
                p_loss = (p_prediction[-1] - p0)**2
        u_loss /= (n+1)
        phi_loss /= (k+1)
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        h_loss = self.mse(h_prediction, self.null)
        self.ls = u_loss + v_loss + 10 * p_loss + 10000 * phi_loss + 0.001 * f_loss + 0.01 * g_loss + 0.01 * h_loss
        self.ls *= 1e-6
        # derivative with respect to net s weights:
        self.ls.backward()
        
    def closure(self):
        # reset gradients to zero:
        self.LBFGS_optimizer.zero_grad()
        x, y, t = next(iter(self.dataloader))
        x = x.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        self.loss_function(x, y, t)
        
        self.iter += 1
        self.ls_average += self.ls.item()
        if self.iter % len(self.dataloader) == 0 :
            self.ls_average /= len(self.dataloader)
            if self.ls_average < min(self.ls_fct_steps) :
                torch.save(self.net.state_dict(), 'autobackup_lbfgs.pt')
            self.ls_fct_steps.append(self.ls_average)
            print('LBFGS EPOCH: {:}, Loss: {:0.6f}'.format(self.iter//len(self.dataloader), self.ls_average))
            self.ls_average = 0
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.3)
        return self.ls

    def LBFGS_train(self):

        # training loop
        self.net.train()
        self.LBFGS_optimizer.step(self.closure)
    
    def Adam_train(self, nb_epoch) :
        min_ls = 1e12
        for i in range(nb_epoch) :
            self.ls_average = 0
            for x, y, t in self.dataloader :
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)
                self.adam_optimizer.zero_grad()
                self.loss_function(x, y, t)
                self.adam_optimizer.step()
                self.iter += 1
                self.ls_average += self.ls.item()
            self.ls_average /= len(self.dataloader)
            if i > nb_epoch // 3  and self.ls_average < min_ls :
                torch.save(self.net.state_dict(), 'autobackup_adam.pt')
                min_ls = self.ls_average
            self.ls_fct_steps.append(self.ls_average)
            print(f'Adam EPOCH {self.iter//len(self.dataloader)}, Loss: {self.ls_average:.6f}')
            
    def train(self, nb_epochs_adam, nb_epoch_lbfgs) :
        self.Adam_train(nb_epochs_adam)
        self.net.load_state_dict(torch.load('autobackup_adam.pt'))
        
        for i in range(nb_epoch_lbfgs) :
            self.LBFGS_train()
        
    def savePINN(self, fileName = 'autobackup.json') :
        torch.save(self.net.state_dict(), 'model' + str(self.iter) + '.pt')
        dico = {}
        dico['net'] = 'model' + str(self.iter) + '.pt'
        dico['loss'] = self.ls.item()
        dico['lossFctSteps'] = self.ls_fct_steps
        dico['iter'] = self.iter
        with open(fileName, 'w') as file :
            json.dump(dico, file, indent=2)
        
    def rollout(self, len_x, len_y,  value = "phi") :
        roll = []
        for x, y, t in self.dataloader :
            x = x.to(self.device)
            y = y.to(self.device)
            t = t.to(self.device)
            u_out, v_out, p_out, phi_out, f_out, g_out, h_out = self.function(x, y, t)
            if value == "phi" :
                u_plot = phi_out.data.cpu().numpy()
            elif value == "u" : 
                u_plot = u_out.data.cpu().numpy()
            elif value == "v" : 
                u_plot = v_out.data.cpu().numpy()
            elif value == "p" : 
                u_plot = p_out.data.cpu().numpy()
            elif value == "f" : 
                u_plot = f_out.data.cpu().numpy()
            elif value == "g" : 
                u_plot = g_out.data.cpu().numpy()
            elif value == "h" : 
                u_plot = h_out.data.cpu().numpy()
            elif value == "x" : 
                u_plot = x.data.cpu().numpy()
            elif value == "y" : 
                u_plot = y.data.cpu().numpy()
            elif value == "t" : 
                u_plot = t.data.cpu().numpy()
            else : return("value don't match")
            u_plot = np.reshape(u_plot, (len_y, len_x))
            roll.append(u_plot)
        return roll
    
    def valid(self) :
        self.ls_average = 0
        for x, y, t in self.dataloader :
            x = x.to(self.device)
            y = y.to(self.device)
            t = t.to(self.device)
            self.loss_function(x, y, t)
            self.ls_average += self.ls
        self.ls_average /= len(self.dataloader)
        return(self.ls_average)
            
            
            
            
            
            
            
            
            
            
            