import random
import numpy as np

class ReplayMemory:

    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.idx = 0
    
    def toElement(self,state,action,next_state,reward,terminal):
        return {'state':state, 'action':action, 'next_state':next_state,'reward':reward,'terminal': 1-int(terminal)}

    def fromElement(self,elm):
        return elm['state'],elm['action'],elm['next_state'],elm['reward'],elm['terminal']
    
    def store(self,elm):
        if(len(self.memory) < self.capacity):
            self.memory += [elm]
        else:
            self.memory[self.idx] = elm
            self.idx = (self.idx + 1) % self.capacity
    
    def sample(self,batchsize):
        indices_to_sample = random.sample(range(len(self.memory)),k = batchsize)

        return np.array(self.memory)[indices_to_sample]

    def __len__(self):
        return len(self.memory)
    
    def getField(self, memory, name = "state"):
        ans = []
        for i in range(len(memory)):    
            ans += [memory[i][name]]
        return np.array(ans)