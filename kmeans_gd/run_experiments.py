import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from models import KMeansSGD, KMeansNR, KMeansLloyd

def run_loss_experiment(datasets, dnames, k, seed=0):
    for ds, dn in zip(datasets, dnames):
        lloyd = KMeansLloyd(k, ds, seed=seed)
        sgd = KMeansSGD(k, ds, seed=seed)
        nr = KMeansNR(k, ds, seed=seed)

        lloyd_losses = np.array(lloyd.fit())
        sgd_losses = np.array(sgd.fit())
        nr_losses = np.array(nr.fit())

        lloyd.view_clusters(name='figures/clusters_{}_lloyd.png'.format(dn))
        sgd.view_clusters(name='figures/clusters_{}_sgd.png'.format(dn))
        nr.view_clusters(name='figures/clusters_{}_nr.png'.format(dn))

        plt.figure()
        plt.title('Losses for {} Dataset'.format(dn))
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.plot(np.arange(len(lloyd_losses)), lloyd_losses, label="Lloyd's algorithm", linestyle='-', color='y', linewidth=2.25)
        plt.plot(np.arange(len(sgd_losses)), sgd_losses, label='gradient descent', linestyle='-', color='r', linewidth=2.25)
        plt.plot(np.arange(len(nr_losses)), nr_losses, label="Newton's method", linestyle=':', color='b', linewidth=2.25)
        plt.legend()
        plt.savefig('figures/loss_{}.png'.format(dn))
        plt.close()


def time_fit(kmeans):
    start_time = time.time()
    kmeans.fit()
    return time.time()-start_time

def run_time_experiment(datasets, k, reps=10, with_nr=True):
    lloyd_dim_time = {d.shape[1]:0 for d in datasets}
    sgd_dim_time = {d.shape[1]:0 for d in datasets}
    if with_nr:
        nr_dim_time = {d.shape[1]:0 for d in datasets}

    for ds in datasets:
        dim = ds.shape[1]
        for i in range(reps):
            lloyd = KMeansLloyd(k, ds, seed=i)
            sgd = KMeansSGD(k, ds, seed=i)
            nr = KMeansNR(k, ds, seed=i)

            lloyd_dim_time[dim] += time_fit(lloyd)/reps
            sgd_dim_time[dim] += time_fit(sgd)/reps
            if with_nr:
                nr_dim_time[dim] += time_fit(nr)/reps

    plt.figure()
    plt.xlabel('dimension')
    plt.ylabel('time (s)')
    plt.title('Fit Time vs Dimension for KMeans Algorithms')
    plt.plot(lloyd_dim_time.keys(),list(lloyd_dim_time.values()), label="Lloyd's algorithm", color='y', linewidth=2.25)

    sgd_line = ':' if with_nr else '-'
    plt.plot(sgd_dim_time.keys(),list(sgd_dim_time.values()), label='gradient descent', color='r', linewidth=2.25, linestyle=sgd_line)
    if with_nr:
        plt.plot(nr_dim_time.keys(),list(nr_dim_time.values()), label="Newton's method", color='b', linewidth=2.25)
    plt.legend()
    endstr = '_noNR' if not with_nr else ''
    plt.savefig('figures/time_dim{}.png'.format(endstr))
    plt.close()

def main():

    d8 = np.loadtxt('data/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)[:,:8]
    d16 = np.loadtxt('data/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)[:,:16]
    d32 = np.loadtxt('data/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)[:,:32]
    d64 = np.loadtxt('data/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)[:,:64]
    d128 = np.loadtxt('data/n1024_d128_k16.txt', delimiter='\t', dtype=np.float32)[:,:128]
    ds = [torch.tensor(ds) for ds in [d8,d16,d32,d64,d128]]

    s1 = np.loadtxt('data/s1.txt', delimiter='\t', dtype=np.float32)
    s2 = np.loadtxt('data/s2.txt', delimiter='\t', dtype=np.float32)
    s3 = np.loadtxt('data/s3.txt', delimiter='\t', dtype=np.float32)
    s4 = np.loadtxt('data/s4.txt', delimiter='\t', dtype=np.float32)
    ss = [torch.tensor(s) for s in [s1,s2,s3,s4]]
    snames = ['s1','s2','s3','s4']

    # 11
    run_loss_experiment(ss, snames, k=15, seed=35)

    # run_time_experiment(ds, k=16, reps=10)
    #run_time_experiment(ds, k=16, reps=10, with_nr=False)










if __name__ == '__main__':
    main()