import gym

import stable_baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'Breakout-v4'
env = gym.make(env_name)

env_name = 'Breakout-v4'
env = gym.make(env_name)
env = gym.wrappers.AtariPreprocessing(env,frame_skip=1)
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, verbose=1)


model.learn(total_timesteps=20000)


model.save('dqn_breakout model')