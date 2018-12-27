import gym
from ddpg import DDPG

def main():
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make('Pendulum-v0')

    # working settings?
    # agent = DDPG(env, sigma=0.1, num_episodes=2000, buffer_size=1000000, batch_size=64, 
    #              tau=1e-2, batch_norm=False, merge_layer=0)

    # paper settings
    # agent = DDPG(env, sigma=0.2, num_episodes=1000, buffer_size=1000000, batch_size=64, 
    #              tau=1e-3, batch_norm=True, merge_layer=2)


    # agent = DDPG(env, sigma=0.2, num_episodes=250, buffer_size=1000000, batch_size=64, 
    #              tau=1e-3, batch_norm=False, merge_layer=0)
    # agent.train()
    # agent.eval(100)


    agent = DDPG(env, sigma=0.2, num_episodes=1000, buffer_size=1000000, batch_size=64, 
                 tau=1e-3, batch_norm=False, merge_layer=0)
    agent.eval_all('models/pendulum')

if __name__ == '__main__':
    main()