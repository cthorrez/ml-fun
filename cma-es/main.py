import numpy as np
import matplotlib.pyplot as plt
from cmaes import CMAES
from problems import LinearRegression, LunarLander


def moving_average(data, n):
    data = np.hstack([data[:n-1], data])
    return np.convolve(data, np.ones(n)/n, 'valid')

def main():

    # run a linear regression test
    # d = 100
    # lin_reg = LinearRegression(d=d)
    # opt = CMAES(d, m=np.random.rand(d), sig=0.1, fitness=lin_reg, log_dir='log/linreg')
    # opt.fit(200)
    # results = np.load('log/linreg/train_results.npy')
    # plt.xlabel('generation')
    # plt.ylabel('loss')
    # plt.title('CMA-ES on Linear Regression')
    # plt.plot(np.arange(len(results)), results)
    # plt.savefig('log/linreg/results.png')
    # # plt.show()


    # run a test on lunar lander
    lander = LunarLander(continuous=True)
    d = lander.mu.dim
    opt = CMAES(d, m=np.random.rand(d), lam=100, sig=0.5, fitness=lander, log_dir='log/lander', save_models=True)
    opt.fit(100)

    results = np.load('log/lander/train_results.npy')*-1
    # multiply by -1 becasue CMA-ES minimized negative reward so this switches it back
    plt.xlabel('generation')
    plt.ylabel('reward')
    plt.title('CMA-ES on LunarLander')
    plt.plot(np.arange(len(results)), results)
    plt.savefig('log/lander/results.png')
    # plt.show()

    


if __name__ == '__main__':
    main()