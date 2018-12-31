import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


class KMeans(nn.Module):
    def __init__(self, k, X, seed=0):
        super(KMeans, self).__init__()
        self.title = ''
        self.X = X
        self.k = k
        self.n ,self.d = X.shape
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.means = self.kmeans_plusplus_initializer()


    def kmeans_plusplus_initializer(self):
        means = torch.zeros((self.k, self.d))
        means[0,:] = self.X[np.random.randint(low=0, high=self.n)]
        for i in range(1, self.k):
            dists = self.get_dists(self.X, means[:i])
            dists, _ = torch.min(dists, dim=1)
            probs = np.clip(dists.numpy(),a_min=0, a_max=np.inf)
            probs = probs/np.sum(probs)

            new_idx = np.random.choice(np.arange(self.n), p=probs)
            means[i,:] = self.X[new_idx]

        return means

    def get_dists(self, X, means):
        # n x 1
        X2 = torch.unsqueeze(torch.sum(torch.pow(X,2), dim=1),1)
        # 1 x k
        means2 = torch.unsqueeze(torch.sum(torch.pow(means,2), dim=1),0)
        # n x k
        Xmeans = torch.mm(X, means.t())
        #n x k
        dists = X2 - 2.0*Xmeans + means2
        return dists

    def update(self, prev_loss):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedErrors

    def get_loss(self):
        dists = self.get_dists(self.X, self.means)
        _, assignments = torch.topk(dists, k=1, largest=False, sorted=True, dim=1)

        assignments = torch.squeeze(assignments)

        approx_X = self.means[assignments]
        loss = torch.sum(torch.pow(self.X - approx_X,2))
        return loss

    def view_clusters(self, name='clusters.png'):
        dists = self.get_dists(self.X, self.means)
        _, assignments = torch.topk(dists, k=1, largest=False, sorted=True, dim=1)
        assignments = torch.squeeze(assignments)

        colors=iter(plt.cm.rainbow(np.linspace(0,1,self.k)))
 
        plt.figure()
        plt.title(self.title)
        for k, c in zip(range(self.k), colors):
            cluster_data = self.X[torch.squeeze(torch.nonzero(assignments==k))]
            if cluster_data.numel() <= self.d : 
                continue
            mx, my = self.means[k,0:2].detach().numpy()
            plt.plot(mx, my, color=c, marker='^', markersize=10, markeredgewidth=1, markeredgecolor='black')

            x = cluster_data[:,0].numpy()
            y = cluster_data[:,1].numpy()
            plt.scatter(x,y, color=c, s=1)
        plt.savefig(name)
        plt.close()


class KMeansSGD(KMeans):
    def __init__(self, k, X, lr = 1e-3, seed=0):
        super(KMeansSGD, self).__init__(k, X, seed=seed)
        self.title = 'Gradient Descent'
        self.means = nn.Parameter(self.means)
        self.initial_lr = self.lr = lr
        self.t = 1


    def update(self, loss):
        grads = torch.autograd.grad(loss, self.means, retain_graph=True, create_graph=True)[0]
        with torch.no_grad():
            self.means.data = self.means.data - self.lr*grads

    def fit(self):
        prev_loss = self.get_loss()
        losses = []
        while True:
            #self.lr = self.lr*0.99
            #self.t += 0

            self.update(prev_loss)
            prev_loss = float(prev_loss)
            losses.append(prev_loss)
            #print(prev_loss)
            new_loss = self.get_loss()
            if float(new_loss) == prev_loss:
                break
            prev_loss = new_loss
        return losses


class KMeansNR(KMeans):
    def __init__(self, k, X, seed=0):
        super(KMeansNR, self).__init__(k, X, seed=seed)
        self.title = "Newton's Method"
        self.means = nn.Parameter(self.means)

    def update(self, loss):
        grads = torch.autograd.grad(loss, self.means, retain_graph=True, create_graph=True)[0]
        hessian = torch.stack([torch.autograd.grad(g, self.means, create_graph=True)[0].view(-1) for g in grads.view(-1)], dim=1)

        # since Hessian is diagonal in this case, inverse is just recripocal of each idag element
        # also since it's diaonal, the matrix vector multiplication is elementwise proct of diagonal and the vector
        delta = ((1/hessian.diag()) * (grads.view(-1))).view(self.k, self.d)

        with torch.no_grad():
            self.means.data = self.means.data - delta

    def fit(self):
        prev_loss = self.get_loss()
        prevprev_loss = np.inf
        losses = []
        while True:
            self.update(prev_loss)
            prev_loss = float(prev_loss)
            # print(prev_loss)
            losses.append(prev_loss)
            new_loss = self.get_loss()
            if float(new_loss) == prev_loss or float(new_loss) == prevprev_loss:
                break
            prevprev_loss = prev_loss
            prev_loss = new_loss
        return losses

class KMeansLloyd(KMeans):
    def __init__(self, k, X, seed=0):
        super(KMeansLloyd, self).__init__(k, X, seed=seed)
        self.title = "Lloyd's Algorithm"

    def update(self, loss):
        dists = self.get_dists(self.X, self.means)
        _, assignments = torch.topk(dists, k=1, largest=False, sorted=True, dim=1)
        assignments = torch.squeeze(assignments)

        for c in range(self.k):
            self.means[c,:] = torch.mean(self.X[assignments==c,:], dim=0)

    def fit(self):
        prev_loss = self.get_loss()
        losses = []
        while True:
            self.update(prev_loss)
            prev_loss = float(prev_loss)
            # print(prev_loss)
            losses.append(prev_loss)
            new_loss = self.get_loss()
            if float(new_loss) == prev_loss:
                break
            prev_loss = new_loss
        return losses






