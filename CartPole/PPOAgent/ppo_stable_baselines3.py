import gym
import stable_baselines3


env = gym.make('CartPole-v1')
model = stable_baselines3.PPO('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps = 25000,n_eval_episodes = 30, eval_log_path="./PPO_evaluation")

model.save('PPO_model')


# Test

print("Testing")
for episode in range(1, 5):
    score = 0
    obs = env.reset()
    done = False
    
    while not done:
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        
    print('Episode:', episode, 'Score:', score)