import gym

import stable_baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'BreakoutNoFrameskip-v4'
env = gym.make(env_name)
env = gym.wrappers.AtariPreprocessing(env,frame_skip=1)
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, buffer_size = 100000, verbose=1)


model.learn(total_timesteps=4000000)


model.save('dqn_breakout 4000000 model')

obs = env.reset

env1 = gym.make(env_name)

while True:
	action = model.predict(obs)
	obs, reward, done, info = env.step(action)
	env1.step(action)
	env1.render()
	if done:
		break
