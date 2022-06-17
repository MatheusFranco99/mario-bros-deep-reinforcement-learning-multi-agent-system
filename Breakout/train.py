import gym

import stable_baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'BreakoutNoFrameskip-v4'
env = gym.make(env_name,render_mode='human')
env = gym.wrappers.AtariPreprocessing(env,frame_skip=1)
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, buffer_size = 100000, verbose=1)


#model.learn(total_timesteps=500)


#model.save('dqn_breakout 4000000 model')

model.load('dqn_breakout 4000000 model')

obs = env.reset()

env1 = gym.make(env_name,render_mode='human')
env1.reset()

while True:
	action, _states = model.predict(obs, deterministic = True)
	print(action)
	obs, reward, done, info = env.step(action)
	env1.step(action)
	print(reward)
	#env1.step(action)
	env1.render(mode='rgb_array')
	if done:
		break
