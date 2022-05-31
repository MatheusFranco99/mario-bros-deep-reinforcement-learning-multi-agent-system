from gettext import npgettext
import os
import numpy as np
import torch
from pettingzoo.atari import mario_bros_v3

from agent import DQNAgent

def fill_memory(env,agent,memory_fill_eps):
    for _ in range(memory_fill_eps):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state):
            next_state,reward,done,info = env.step(action)
            agent.replay_memory.store(state,action,next_state,reward,done)
            state = next_state
    

def train(env,agent,train_eps,memory_fill_eps,batchsize,update_freq,model_filename):
    fill_memory(env,agent,memory_fill_eps)
    print('Samples in memory:',len(agent.replay_memory))

    step_cnt = 0
    reward_history = []
    best_score = -np.inf

    for ep_cnt in range(train_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state,reward,done,info = env.step(action)
            agent.replay_memory.store(state,action,next_state,reward,done)

            agent.learn(batchsize)

            if step_cnt % update_freq == 0:
                agent.update_target()
            
            state = next_state
            ep_reward += reward
            step_cnt += 1
        
        agent.update_epsilon()
        reward_history.append(ep_reward)

        current_avg_score = np.mean(reward_history[-100:])

        print(f'Ep:{ep_cnt}, Total Steps:{step_cnt}, Ep Score: {ep_reward}, Avg score: {current_avg_score}; Updated Epsilon {agent.epsilon}')

        if current_avg_score >= best_score:
            agent.save(model_filename)
            best_score = current_avg_score
        

def test(env,agent,test_eps):
    for ep_cnt in range(test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
        
        print(f"Ep: {ep_cnt}, Ep score: {ep_reward}")
        

def set_seed(env,seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    