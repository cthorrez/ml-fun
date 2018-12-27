import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()

    def initialize_weights(self):
        num_layers = len(list(self.children()))
        for l in self.layers[:-1]:
            f = l.weight.shape[1]
            bound = 1/np.sqrt(f)
            nn.init.uniform_(l.weight, -bound, bound)
            nn.init.uniform_(l.bias, -bound, bound)

        nn.init.uniform_(self.layers[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layers[-1].bias, -3e-3, 3e-3)



class Actor(Network):
    def __init__(self, s_dim, a_dim, action_space, n_hid=(400,300), batch_norm=False):
        super(Actor,self).__init__()
        self.ac_low = torch.squeeze(torch.tensor(action_space.low))
        self.ac_high = torch.squeeze(torch.tensor(action_space.high))
        self.batch_norm = batch_norm
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        dims = [s_dim, *n_hid, a_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(n_hid)+1)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(d) for d in dims[:-1]])
        self.initialize_weights()

    def forward(self, s):
        if len(s.shape) == 1 : s = torch.unsqueeze(s, 0)
        x = s
        for layer, bn in zip(self.layers[:-1], self.batch_norms[:-1]):
            if self.batch_norm : x = bn(x)
            x = layer(x)
            x = self.relu(x)
        if self.batch_norm : x = self.batch_norms[-1](x)
        x = self.layers[-1](x)
        x = self.tanh(x)
        x = (x + (self.ac_high+self.ac_low)/2) * ((self.ac_high - self.ac_low)/2)
        return x

class Critic(Network):
    def __init__(self, s_dim, a_dim, n_hid=(400,300), batch_norm=False, merge_layer=2):
        super(Critic,self).__init__()
        self.relu = nn.ReLU()
        self.merge_layer = merge_layer

        out_dims = [*n_hid, 1]
        in_dims = [s_dim, *n_hid]
        in_dims[self.merge_layer] += a_dim
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList([])
        for i_d, o_d in zip(in_dims, out_dims):
            self.layers.append(nn.Linear(i_d, o_d))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(d) for d in in_dims[:2]])
        self.initialize_weights()

    def forward(self, s, a):
        if len(s.shape) == 1 : s = torch.unsqueeze(s, 0)
        x = s
        for idx, layer in enumerate(self.layers):
            if idx == self.merge_layer:
                x = torch.cat([x,a], dim=1)
            if (idx < self.merge_layer) and self.batch_norm:
                x = self.batch_norms[idx](x)

            x = layer(x)
            if idx < len(self.layers)-1:
                x = self.relu(x)
        return x


