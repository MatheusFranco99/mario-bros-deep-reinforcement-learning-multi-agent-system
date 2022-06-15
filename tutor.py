import supersuit as ss
from stable_baselines3 import PPO, DQN
# from stable_baselines3.deepq.policies import CnnPolicy

import gym
import flappy_bird_gym

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
env = gym.make("SpaceInvaders-v4",full_action_space = False)
env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:195,:,:], lambda obs_space:obs_space)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=64, y_size=64)
#env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
env = ss.action_lambda_v1(env,lambda action, act_space : action, lambda act_space : gym.spaces.Discrete(6))
env = ss.frame_stack_v1(env, 4)
#env = ss.black_death_v3(env)
#env = ss.gym_env_to_vec_env_v1(env)
#env = vectorize_aec_env_v0(env)
env = ss.concat_vec_envs_v1(env, 32, num_cpus=1, base_class="stable_baselines3")



model = DQN("MlpPolicy", env, verbose=1,buffer_size=100000)

model.learn(total_timesteps=30000000)
model.save("spaceInvadersSingle_6464_6actions_30000000")

# Rendering

#env = space_invaders_v2.env()
env = gym.make("SpaceInvaders-v4",render_mode='human',full_action_space = False)
env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:195,:,:], lambda obs_space:obs_space)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=64, y_size=64)
# env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
env = ss.action_lambda_v1(env,lambda action, act_space : action, lambda act_space : gym.spaces.Discrete(6))
env = ss.frame_stack_v1(env, 4)

model = DQN.load("spaceInvadersSingle_6464_6actions_30000000")

env2 = gym.make("SpaceInvaders-v4",render_mode='human',full_action_space = False)
obs = env.reset()
env2.reset()
env2.render(mode='rgb_array')
done = False
# for agent in env.agent_iter():
total = 0
while True:
    # obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    obs, reward, done, info = env.step(act)
    env2.step(act)
    total += reward

    env2.render(mode='rgb_array')
    if(done):
        break
