import numpy as np
import torch
from models import KMeansSGD, KMeansNR, KMeansLloyd
import shutil, os
import multiprocessing
import copy
from functools import partial


# fits one kmeans model
def fit_kmeans(data, k, seed, max_iter=1000):
    
    model = KMeansNR(X=data, k=k, seed=seed)
    #model = KMeansSGD(X=data, k=k, seed=seed)
    #model = KMeansLloyd(X=data, k=k, seed=seed)

    model.fit()
    return model

# fits many kmeans models via multiprocessing
# uses the one with the lowest loss
def main():
    n_init = 20
    n_jobs = 2

    if os.path.exists('figures'):
        shutil.rmtree('figures')
    os.mkdir('figures')

    k = 15
    data = np.loadtxt('../data/cluster/n5000_d2_k15.txt', delimiter='\t', dtype=np.float32)

    # k = 16
    #data = np.loadtxt('../data/cluster/n1024_d32_k16.txt', delimiter='\t', dtype=np.float32)
    #data = np.loadtxt('../data/cluster/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)
    #data = np.loadtxt('../data/cluster/n1024_d256_k16.txt', delimiter='\t', dtype=np.float32)
    
    data = torch.tensor(data)

    func = partial(fit_kmeans, data, k)
    pool = multiprocessing.Pool(n_jobs)
    seeds = np.arange(n_init)
    models = pool.map(func, seeds)

    losses = [float(m.get_loss()) for m in models]
    best_model = models[np.argmin(losses)]
    best_model.view_clusters()
    print('loss:', np.min(losses))

if __name__ == '__main__':
    main()