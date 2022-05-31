import random
import torch
import torch.nn.functional as F
from replay_memory import ReplayMemory
from dqn import DQNNet

class DQNAgent:
    def __init__(self,observation_space,action_space,device,epsilon_max,epsilon_min,epsilon_decay,memory_capacity,discount=0.9,lr=1e-3):
        self.obervation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.device = device
        
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_memory = ReplayMemory(memory_capacity)

        self.online_network = DQNNet(self.obervation_space.shape[0],self.action_space.n,lr).to(self.device)
        self.target_network = DQNNet(self.obervation_space.shape[0],self.action_space.n,lr).to(self.device)

        self.target_network.eval()
        self.update_target()
    
    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def select_action(self,state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action = torch.argmax(self.online_network(state))
        return action.item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)
    
    def learn(self,batchsize):
        if(len(self.replay_memory) < batchsize):
            return
        
        states,actions,next_states,rewards,dones = self.replay_memory.sample(batchsize,self.device)

        actions = actions.reshape((-1,1))
        rewards = rewards.reshape((-1,1))
        dones = dones.reshape((-1,1))

        predicted_qs = self.online_network(states)
        predicted_qs = predicted_qs.gather(1,actions)

        target_qs = self.target_network(next_states)
        target_qs = torch.max(target_qs,dim=1).values
        target_qs = target_qs.reshape(-1,1)
        target_qs[dones] = 0.0

        y_js = rewards + (self.discount * target_qs)

        loss = F.mse_loss(predicted_qs,y_js)
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
    
    def save(self,filename):
        torch.save(self.online_network.state_dict(),filename)
    
    def load(self,filename):
        self.online_network.load_state_dict(torch.load(filename))
        self.online_network.eval()