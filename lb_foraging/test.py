import lbforaging
import gym

env = gym.make("Foraging-8x8-2p-1f-v2")

env.reset()

nobs,nrewards,ndone,ninfo = env.step([0,0])

