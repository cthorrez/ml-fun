import numpy as np
import pickle
import gym
import time

def demo_ddpg_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    mu = pickle.load(open('log/lander/models/model_75','rb'))
    while True:
        a = mu(s)
        s, r, done, _ = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            pass
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            # print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        if done:
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    time.sleep(3)

    for i in range(10):
        demo_ddpg_lander(env, seed=i, render=True)
        time.sleep(0.5)