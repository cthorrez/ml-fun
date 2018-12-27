import gym
from ddpg import DDPG

def main():
    #env = gym.make('LunarLanderContinuous-v2')
    #log_dir = 'log/lander'

    env = gym.make('Pendulum-v0')
    log_dir = 'log/pendulum'

    # paper settings
    # agent = DDPG(env, sigma=0.2, num_episodes=1000, buffer_size=1000000, batch_size=64, 
    #              tau=1e-3, batch_norm=True, merge_layer=2)

    # did not work unless I merged action into critic at first layer
    # worked btter without batchnorm
    agent = DDPG(env, sigma=0.2, num_episodes=250, buffer_size=1000000, batch_size=64, 
                 tau=1e-3, batch_norm=False, merge_layer=0)
    agent.train()
    agent.eval_all(log_dir+'/models', num_eps=10)

if __name__ == '__main__':
    main()