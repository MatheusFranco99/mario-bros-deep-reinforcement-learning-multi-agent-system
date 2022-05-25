from time import sleep
from pettingzoo.atari import mario_bros_v3


env = mario_bros_v3.env()

height, width, channels = env.observation_space.shape
actions = env.action_space.n
