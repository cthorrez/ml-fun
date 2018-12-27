from networks import Actor, Critic
from noise import OrnsteinUhlenbeck
from buffer import Buffer
import torch
import numpy as np
import copy
import os

class DDPG():
    def __init__(self, env, log_dir, gamma=0.99, batch_size=64, sigma=0.2, batch_norm=True, merge_layer=2,
                 buffer_size=int(1e6), buffer_min=int(1e4), tau=1e-3, Q_wd=1e-2, num_episodes=1000):

        self.s_dim = env.reset().shape[0]
        self.a_dim = env.action_space.shape[0]

        self.env = env
        self.mu = Actor(self.s_dim, self.a_dim, env.action_space, batch_norm=batch_norm)
        self.Q = Critic(self.s_dim, self.a_dim, batch_norm=batch_norm, merge_layer=merge_layer)
        self.targ_mu = copy.deepcopy(self.mu).eval()
        self.targ_Q = copy.deepcopy(self.Q).eval()
        self.noise = OrnsteinUhlenbeck(mu=torch.zeros(self.a_dim), sigma=sigma * torch.ones(self.a_dim))
        self.buffer = Buffer(buffer_size, self.s_dim, self.a_dim)
        self.buffer_min = buffer_min
        self.mse_fn = torch.nn.MSELoss()
        self.mu_optimizer = torch.optim.Adam(self.mu.parameters(), lr=1e-4)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3, weight_decay=Q_wd)

        self.gamma = gamma
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.tau = tau

        self.fill_buffer()

    #updates the target network to slowly track the main network
    def track_network(self, target, main):
        with torch.no_grad():
            for pt, pm in zip(target.parameters(), main.parameters()):
                pt.data.copy_(self.tau*pm.data + (1-self.tau)*pt.data)
                


    # updates the target nets to slowly track the main ones
    def track_networks(self):
        self.track_network(self.targ_mu, self.mu)
        self.track_network(self.targ_Q, self.Q)

    def run_episode(self):
        done = False
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)
        t = 0
        tot_r = 0
        while not done:

            self.mu = self.mu.eval()
            a = torch.squeeze(self.mu(s)).detach().numpy()
            self.mu = self.mu.train()


            ac_noise = self.noise().detach().numpy()
            a = a + ac_noise

            s = s.detach().numpy()
            s_p, r, done, _ = self.env.step(a)
            tot_r += r
            self.buffer.add_tuple(s,a,r,s_p,done)
  
            s_batch, a_batch, r_batch, s_p_batch, done_batch = self.buffer.sample(batch_size=self.batch_size)

            # update critic
            with torch.no_grad():
                q_p_pred = self.targ_Q(s_p_batch, self.targ_mu(s_p_batch))
                q_p_pred = torch.squeeze(q_p_pred)
                y = r_batch + (1.0 - done_batch)*self.gamma*q_p_pred
            self.Q_optimizer.zero_grad()
            q_pred = self.Q(s_batch, a_batch)
            q_pred = torch.squeeze(q_pred)
            #print(torch.mean(q_pred))
            Q_loss = self.mse_fn(q_pred, y)
            Q_loss.backward(retain_graph=False)
            self.Q_optimizer.step()

            # update actor
            self.mu_optimizer.zero_grad()
            q_pred_mu = self.Q(s_batch, self.mu(s_batch))
            q_pred_mu = torch.squeeze(q_pred_mu)
            #print(torch.mean(q_pred_mu))
            mu_loss = -torch.mean(q_pred_mu)
            # print(mu_loss)
            mu_loss.backward(retain_graph=False)
            #print(torch.sum(self.mu.layers[0].weight.grad))
            self.mu_optimizer.step()
            self.track_networks()

            s = torch.tensor(s_p.astype(np.float32), requires_grad=False)
            t += 1
        return tot_r, t

    def train(self):
        results = []
        for i in range(self.num_episodes):
            r, t = self.run_episode()
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
            results.append([r,t])

            if i % 5 == 0:
                torch.save(self.mu, self.log_dir +'/models/model_'+str(i))
        np.save(self.log_dir + '/results_train.npy', np.array(results))


    def run_eval_episode(self, mu=None):
        if mu == None:
            mu = self.mu
        done = False
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)
        tot_r = t = 0
        while not done:
            a = mu(s).view(-1).detach().numpy()

            s_p, r, done, _ = self.env.step(a)
            tot_r += r
            t += 1
            s = torch.tensor(s_p.astype(np.float32), requires_grad=False)
        return tot_r, t

    def eval_all(self, model_dir):
        results = []
        for model_fname in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[2])):
            print(model_fname)
            mu = torch.load(os.path.join(model_dir, model_fname))
            r,t = self.eval(num_eps=10, mu=mu)
            results.append([r,t])
        np.save(self.log_dir+'/results_eval.npy', np.array(results))

    def eval(self, num_eps=100, mu=None):
        if mu == None:
            mu = self.mu

        results = []
        mu = mu.eval()
        for i in range(num_eps):
            r, t = self.run_eval_episode(mu=mu)
            results.append([r,t])
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
        return np.mean(results, axis=0)

    def fill_buffer(self):
        print('Filling buffer')
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)
        while self.buffer.size < self.buffer_min:
            a = np.random.uniform(self.env.action_space.low, self.env.action_space.high, size=(self.a_dim))

            s_p, r, done, _ = self.env.step(a)
            if done:
                self.env.reset()
            self.buffer.add_tuple(s,a,r,s_p,done)
            s = s_p




