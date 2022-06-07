import supersuit as ss
from stable_baselines3 import PPO, DQN
# from stable_baselines3.deepq.policies import CnnPolicy

import gym

import numpy as np

from pettingzoo.atari import mario_bros_v3
import matplotlib.pyplot as plt

def actChange(a):
    if a == 0:
        return 0 # stay
    elif a == 1:
        return 1 # jump
    elif a == 2:
        return 3 # right
    elif a == 3:
        return 4 # left
    elif a == 4:
        return 11 # jump right
    elif a == 5:
        return 12 # jump left
    elif a == 6:
        return 13 # jump higher

env = mario_bros_v3.env()
env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:175,:,:], lambda obs_space:obs_space)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=64, y_size=64)
env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
env = ss.frame_stack_v1(env, 4)
env.reset()

k = 0
action = 0

score = {'first_0':0,'second_0':0}

for agent in env.agent_iter():
    # k = k + 1
    # if (k%80 == 0):
    #     action = int(input())
    #     k = 0
    o,r,d,i = env.last()
    score[agent] += r
    if(d):
        env.step(None)
        print("Done, reward:",r)
    else:
    # env.step(action)
        env.step(np.random.choice(list(range(7)),1)[0])

    
    print(f"\rScores {score['first_0']}, {score['second_0']}",end="\r")


    if(r != 0 or len(i) != 0):
        print(f"{r=},{i=}")

    env.render()

# plt.imshow(o)

# env = ss.black_death_v3(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")

# 0 parado
# 1 pula
# 3 direita
# 4 esquerda
# 11 pula pra direita
# 12 pula pra esquerda
# 13 pula mais alto


# tirar 2,5,6,7,8,9,10,14,15,17