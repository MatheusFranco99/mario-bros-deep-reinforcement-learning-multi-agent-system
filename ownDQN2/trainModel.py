
import numpy as np
import colorama



def learn_step(agent,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    action = None
    if(done):
        action = None
        env.step(action)
    else:
    
        action = agent.choose_action(np.array(state))
        # print(action)
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name] - 1
            observation = env.observe(name)

            old_state = np.array(state.copy())

            state.pop(0)
            state+=[observation]
            score+=reward
            agent.remember(old_state,action,np.array(state),reward,done)
            agent.learn()

    return agent,state,action,score,done,env

def dont_learn_step(agent,action,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    if(done):
        action = None
        env.step(None)
    else:
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name] - 1
            observation = env.observe(name)

            old_state = np.array(state.copy())

            state.pop(0)
            state+=[observation]
            score+=reward
            # if(reward!=0):
            agent.remember(old_state,action,np.array(state),reward,done)

    return agent,state,action,score,done,env

def progress_bar(progress, total, color = colorama.Fore.YELLOW):
    percent = int(100*(progress / float(total)))
    bar = chr(9608) * percent + '-' * (100-percent)
    if (progress == total):
        color = colorama.Fore.GREEN
    print(color + f"\r|{bar}| {percent:.2f}%", end='\r')

def trainModels(agent1,agent2,env,num_max_steps=200000,n_epochs = 100,jumpKSteps=3,render=False,):


    scores=[]

    for epoch in range(n_epochs):

        env.reset()
        if render:
            env.render()
        state, r, d, info = env.last()

        score1 = 0
        score2 = 0
        state1 = []
        state2 = []
        done1 = False
        done2 = False
        for i in range(4):
            o,r,done1,i = env.last()
            state1+=[o]
            score1 += r
            env.step(0)
            o,r,done2,i = env.last()
            state2+=[o]
            score2+=r
            env.step(0)

        action1 = 0
        action2 = 0

        done = done1 and done2


        for i in range(num_max_steps):
            if(i%1000 == 0):
                print(f"\r{epoch=},steps={i}",end="\r")
            # progress_bar(i,num_max_steps)

            if not done:
                if not done1:
                    agent1,state1,action1,score1,done1,env = learn_step(agent1,state1,score1,env)
                    if render:
                        env.render()
                if not done2:
                    agent2,state2,action2,score2,done2,env = learn_step(agent2,state2,score2,env,name='second_0')
                    if render:
                        env.render()

                for j in range(jumpKSteps):
                    if not done1:
                        agent1,state1,action1,score1,done1,env = dont_learn_step(agent1,action1,state1,score1,env)
                        if render:
                            env.render()
                    if not done2:
                        agent2,state2,action2,score2,done2,env = dont_learn_step(agent2,action2,state2,score2,env,name='second_0')
                        if render:
                            env.render()
                
                done = done1 and done2
            else:
                break

        progress_bar(num_max_steps,num_max_steps)
        print()
        
        scores += [[score1,score2]]

        print(f"Ended epoch {epoch} with scores: {score1}, {score2}")
    
    agent1.save_model()
    agent2.save_model()



