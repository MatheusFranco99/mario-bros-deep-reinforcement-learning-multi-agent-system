
import numpy as np
import colorama

def progress_bar(progress, total, reward, color = colorama.Fore.YELLOW):
    percent = int(100*(progress / float(total)))
    bar = chr(9608) * percent + '-' * (100-percent)
    if (progress == total):
        color = colorama.Fore.GREEN
    print(color + f"\r|{bar}| {percent:.2f}%, last reward: {reward}", end='\r')

def train(agent,env,n_episodes = 1000, c = 100, render = False):

    scores=[]

    for episode in range(n_episodes):

        state = env.reset()
        if render:
            env.render()

        step = 0
        total_reward = 0
        while True:
            step += 1

            if(np.random.random() < agent.epsilon):
                action = agent.pickRandomAction()
            else:
                action = agent.pickAction()
            
            next_state,reward,done = env.step(action)
            total_reward += reward
            agent.store(state,action,next_state,reward,done)

            agent.learn()
            
            if step%c == 0:
                step = 0
                agent.updateQ()
            
            if done:
                scores.append(total_reward)
                break


        progress_bar(episode,n_episodes,total_reward)

    agent.save_model()
    return agent



