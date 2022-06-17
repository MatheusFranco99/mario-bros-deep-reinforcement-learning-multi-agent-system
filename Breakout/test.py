import gym
import time
env = gym.make("Breakout-v4",render_mode="human")
env.reset()
env.render(mode="rgb_array")
time.sleep(10)
