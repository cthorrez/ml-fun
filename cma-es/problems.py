import numpy as np
import gym
import pickle
from multiprocessing import Pool
from networks import Actor

class Problem:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError



class LinearRegression(Problem):
    def __init__(self, n=100, d=3, sigma=10):
        super(LinearRegression,self).__init__()

        self.beta = np.random.rand(d)

        self.X = np.random.rand(n,d-1)
        self.X = np.hstack([np.ones((n,1)), self.X])
        noise = np.random.randn(n)*sigma
        self.y = np.dot(self.X,self.beta) + noise


        beta_hat = np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T,self.X)),self.X.T),self.y)
        y_pred = np.dot(self.X,beta_hat)
        print('min loss:', np.mean(np.power(y_pred-self.y,2)))


    def score(self, beta):
        y_pred = np.dot(self.X,beta)
        return np.mean(np.power(y_pred-self.y,2))

    def score_vec(self, betas):
        return [self.score(beta) for beta in betas]


class LunarLander(Problem):
    def __init__(self, continuous=True, seed=0):
        super(LunarLander,self).__init__()
        if continuous:
            self.env = gym.make('LunarLanderContinuous-v2')
        else:
            self.env = gym.make('LunarLander-v2')
        self.env.seed(seed)
        self.mu = Actor(s_dim=8, a_dim=4, h_dim=20)


    def score(self, x):
        self.mu.fill_weights(x)
        r_tot = 0.
        done = False
        s = self.env.reset()
        while not done:
            a = self.mu(s)
            s, r, done, _ = self.env.step(a)
            r_tot += r
        return -r_tot

    def score_vec(self, X):
        p = Pool(4)
        return p.map(self.score, [x for x in X])
        
    def save(self, x, g):
        self.mu.fill_weights(x)
        pickle.dump(self.mu, open('log/lander/models/model_'+str(g),'wb'))














