import supersuit as ss
from stable_baselines3 import PPO, DQN
# from stable_baselines3.deepq.policies import CnnPolicy


from pettingzoo.atari import mario_bros_v3

# env = mario_bros_v3.parallel_env()
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.resize_v1(env, x_size=40, y_size=40)
# env = ss.frame_stack_v1(env, 10)
# env = ss.black_death_v3(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")



# model = DQN("CnnPolicy", env, verbose=1)

# model.learn(total_timesteps=2000000)
# model.save("policy4040_10_2000000")

# Rendering

env = mario_bros_v3.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=40, y_size=40)
env = ss.frame_stack_v1(env, 10)

model = DQN.load("policy4040_10_2000000")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()