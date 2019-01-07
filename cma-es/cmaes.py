import numpy as np
import math
from networks import Actor
import shutil
import os

class CMAES:
    def __init__(self, n, m, sig, fitness, lam=None, mu=None, alph_cov=2., c_m=1., c_s=None, c_c=None, 
                 c_1=None, c_mu=None, d_s=None, log_dir='log/default', save_models=False, seed=0):

        np.random.seed(seed)
        self.log_dir = log_dir
        self.save_models = save_models
        if self.save_models:
            shutil.rmtree(log_dir+'/models')
            os.mkdir(log_dir+'/models')

        self.n = n
        self.m = m
        self.sig = sig
        self.fitness = fitness
        self.lam = lam
        self.mu = mu

        self.alph_cov = alph_cov
        self.c_c = c_c
        self.c_1 = c_1
        self.c_s = c_s
        self.c_m = c_m
        self.c_mu = c_mu
        self.d_s = d_s
    
        self.p_s = np.zeros(n)
        self.p_c = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        # expected value of norm of n dimensional Gaussian with identity covariance
        # I think this is a second order Taylor approximation or something?
        self.E_NN0I = math.sqrt(n)*(1.-(1./(4*n))+(1./(21.*math.pow(n,2))))

        if self.lam == None:
            self.lam = 4 + math.floor(3. * math.log(self.n))

        if self.mu == None:
            self.mu = math.floor(self.lam/2.)

        w_p = np.array([math.log((self.lam + 1.)/2.)-math.log(i) for i in range(1,self.lam+1)])

        self.mu_eff = np.power(np.sum(w_p[:self.mu]),2) / np.sum(np.power(w_p[:self.mu],2))
        mu_eff_neg = np.power(np.sum(w_p[self.mu:]),2) / np.sum(np.power(w_p[self.mu:],2))

        if self.c_s == None:
            self.c_s = (self.mu_eff + 2.) / (self.n + self.mu_eff +5.)

        if self.d_s == None:
            self.d_s = 1. + 2*max(0., math.sqrt((self.mu_eff-1.)/(self.n+1.))-1.) + self.c_s

        if self.c_c == None:
            self.c_c = (4.+(self.mu_eff/self.n))/(self.n+4.+(2*self.mu_eff)/self.n)

        if self.c_1 == None:
            self.c_1 = self.alph_cov / (math.pow(self.n+1.3,2) + self.mu_eff)

        if self.c_mu == None:
            rhs = (self.alph_cov*(self.mu_eff-2.+(1./self.mu_eff))) / (math.pow(self.n+2.,2)+(self.alph_cov*self.mu_eff)/2.)
            self.c_mu = min(1.-self.c_1, rhs)

        alph_mu = 1. + (self.c_1/self.c_mu)
        alph_mu_eff = 1. + ((2.*mu_eff_neg)/(self.mu_eff+2.))
        alph_pos_def = (1.-self.c_1-self.c_mu)/(self.n*self.c_mu)

        self.w = np.zeros(self.lam)
        for i in range(self.lam):
            if w_p[i] >= 0:
                self.w[i] = w_p[i] / np.sum(np.abs(w_p*(w_p >= 0)))
            else:
                self.w[i] = (w_p[i] * np.min([alph_mu, alph_mu_eff, alph_pos_def])) / np.sum(np.abs((w_p * (w_p<0))))

    def fit(self, num_gen):
        results = []
        for g in range(num_gen+1):
            X = np.empty((self.lam, self.n))
            Y = np.empty((self.lam, self.n))
            for i in range(self.lam):
                z = np.random.randn(self.n) 
                y =  np.dot(self.B, self.D*z)
                x = self.m + self.sig*y
                X[i,:] = x 
                Y[i,:] = y

            # fits = np.array([self.fitness.score(x) for x in X])
            fits = np.array(self.fitness.score_vec(X))
            idxs = np.argsort(fits)

            y_mean = np.zeros(self.n)
            for j in range(self.mu):
                y_mean += self.w[j] * Y[idxs[j],:]

            # update mean
            self.m = self.m + self.c_m*self.sig*y_mean

            mean_score = self.fitness.score(self.m)
            print(mean_score)
            results.append(mean_score)
            if self.save_models and ((g % 5) == 0):
                self.fitness.save(self.m, g)

            # stepsize control
            C_inv_2 = np.matmul(self.B, np.diagflat(self.D))
            self.p_s = (1-self.c_s)*self.p_s + np.sqrt(self.c_s*(2-self.c_s)*self.mu_eff)*np.dot(C_inv_2,y_mean)
            self.sig = self.sig*math.exp((self.c_s/self.d_s)*((np.linalg.norm(self.p_s)/self.E_NN0I)-1))

            # covariance matrix adaption
            h_sig = 0.
            if np.linalg.norm(self.p_s)/(math.sqrt(1-math.pow(1-self.c_s,2*(g+1)))) < (1.4+(2./(self.n+1)))*self.E_NN0I:
                h_sig = 1.


            self.p_c = (1-self.c_c)*self.p_c + h_sig*math.sqrt(self.c_c*(2-self.c_c)*self.mu_eff)*y_mean

            mask = np.ones(self.lam)
            for i in range(self.mu, self.lam):
                mask = (self.n)/(math.pow(np.linalg.norm(np.dot(C_inv_2,Y[idxs[i],:])),2))
            w_cir = self.w * mask

            delt_h_sig = int((1-h_sig)*self.c_c*(2-self.c_c) <= 1)

            y_tmp = Y[idxs,:]
            self.C = (1+self.c_1*delt_h_sig-self.c_1-self.c_mu*np.sum(self.w))*self.C \
                     + self.c_1*np.matmul(self.p_c[:,None],self.p_c[:,None].T) \
                     + self.c_mu * np.matmul(np.matmul(y_tmp.T, np.diagflat(w_cir)), y_tmp)

            D2, self.B = np.linalg.eig(self.C)
            # D2, self.B = np.linalg.eigh(self.C) # should be faster and work becasue C should be symmetric
            self.D = np.sqrt(D2)

        np.save(self.log_dir+'/train_results.npy',np.array(results))










