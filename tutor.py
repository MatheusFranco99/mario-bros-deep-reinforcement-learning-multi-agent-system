import supersuit as ss
from stable_baselines3 import PPO, DQN
# from stable_baselines3.deepq.policies import CnnPolicy

import gym

from pettingzoo.atari import mario_bros_v3,space_invaders_v2

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

env = space_invaders_v2.parallel_env()
env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:195,:,:], lambda obs_space:obs_space)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=64, y_size=64)
# env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
env = ss.frame_stack_v1(env, 4)
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 32, num_cpus=1, base_class="stable_baselines3")



model = DQN("CnnPolicy", env, verbose=1,buffer_size=100000)

model.learn(total_timesteps=200000)
model.save("spaceInvaders_6464_200000")

# Rendering

env = space_invaders_v2.env()
env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:195,:,:], lambda obs_space:obs_space)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=64, y_size=64)
# env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
env = ss.frame_stack_v1(env, 4)

model = DQN.load("spaceInvaders_6464_200000")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
