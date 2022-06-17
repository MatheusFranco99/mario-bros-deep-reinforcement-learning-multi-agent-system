import numpy as np

from Agent import Agent

def testStep(agent,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    # if r < 0:
    #     print(f"Neg Reward: {r}")
    action = None
    if(done):
        action = None
        env.step(action)
    else:
        action = agent.choose_action(np.array(state),deterministic=True)
        # print(action)
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name]
            observation = env.observe(name)

            state.pop(0)
            state+=[observation]
            score+=reward

    return agent,state,action,score,done,env




def testModels(agent1,agent2,env,render=False,num_max_steps=10000):

    # agent1 = Agent(height=64,width=64,num_fraces=4,n_actions=7,epsilon=1,fname=f"{fname}_a1.h5")
    # agent1.load_model()
    # agent2 = Agent(height=64,width=64,num_fraces=4,n_actions=7,epsilon=1,fname=f"{fname}_a2.h5")
    # agent2.load_model()



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
        if not done:
            if not done1:
                agent1,state1,action1,score1,done1,env = testStep(agent1,state1,score1,env)
                if render:
                    env.render()
            if not done2:
                agent2,state2,action2,score2,done2,env = testStep(agent2,state2,score2,env,name='second_0')
                if render:
                    env.render()
            
            done = done1 and done2
    
    print(f"{score1=}{score2=}")

