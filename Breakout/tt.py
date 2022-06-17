from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C



env = make_atari_env('BreakoutNoFrameskip-v4', n_envs = 16)

env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy",env,verbose=1)
model.learn(total_timesteps=int(5e6))

model.save("A2C_breakout")
obs = env.reset()
while True:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()