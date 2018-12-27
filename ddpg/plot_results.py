import numpy as np
import matplotlib.pyplot as plt




def moving_average(data, n):
    data = np.hstack([data[:n-1], data])
    return np.convolve(data, np.ones(n)/n, 'valid')





def main():
    # plot pendulum results
    train = np.load('log/pendulum/results.npy')
    x = np.arange(train.shape[0])
    y = train[:,0]
    y = moving_average(y, 25)
    plt.plot(x, y, label='train')

    evaluation = np.load('log/pendulum/results_eval.npy')
    x = np.arange(evaluation.shape[0])*5
    y = evaluation[:,0]
    #y = moving_average(y, 1)
    plt.plot(x, y, label='evaluation')

    plt.title('DDPG on Pendulum-v0')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()


    # plot lunar lander results
    train = np.load('log/lander/results.npy')
    x = np.arange(train.shape[0])
    y = train[:,0]
    y = moving_average(y, 100)
    plt.plot(x, y, label='train')

    evaluation = np.load('log/lander/results_eval.npy')
    x = np.arange(evaluation.shape[0])*25
    y = evaluation[:,0]
    y = moving_average(y, 4)
    plt.plot(x, y, label='evaluation')

    plt.title('DDPG on LunarLanderContinuous-v2')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()