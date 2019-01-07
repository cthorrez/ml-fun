import numpy as np


def relu(x):
    return x*(x>0)


class Actor:
    def __init__(self, s_dim, a_dim, h_dim=20):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.w1 = np.empty((s_dim,h_dim))
        self.b1 = np.empty(h_dim)
        self.w2 = np.empty((h_dim,a_dim))
        self.b2 = np.empty(a_dim)
        self.dim = np.sum([x.size for x in [self.w1,self.b1,self.w2,self.b2]])
        print('Network parameters:', self.dim)

    def fill_weights(self, x):
        s_dim, a_dim, h_dim = self.s_dim, self.a_dim, self.h_dim
        cutoffs = [0,s_dim*h_dim, s_dim*h_dim+h_dim, s_dim*h_dim+h_dim+h_dim*a_dim, self.dim]
        self.w1 = np.reshape(x[cutoffs[0]:cutoffs[1]], (s_dim,h_dim))
        self.b1 = x[cutoffs[1]:cutoffs[2]]
        self.w2 = np.reshape(x[cutoffs[2]:cutoffs[3]], (h_dim,a_dim))
        self.b2 = x[cutoffs[3]:cutoffs[4]]

    def __call__(self, s):
        a1 = relu(np.dot(s,self.w1) + self.b1)
        a2 =  np.dot(a1,self.w2) + self.b2
        return a2

        




