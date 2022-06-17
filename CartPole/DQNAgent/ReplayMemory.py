import random

class ReplayMemory:

    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.idx = 0
        
    def store(self,state,action,next_state,reward,done):
        elm = (state,action,next_state,reward,done)
        if(len(self.memory) < self.capacity):
            self.memory.append(elm)
        else:
            self.memory[self.idx] = elm
            self.idx = (self.idx + 1) % self.capacity
    
    def sample(self,batchsize):
        return random.sample(self.memory,batchsize)

    def __len__(self):
        return len(self.memory)
