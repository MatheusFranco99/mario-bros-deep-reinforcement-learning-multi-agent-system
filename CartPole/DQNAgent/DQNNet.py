from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

class DQNNet:

    def __init__(self,observation_dim, num_actions,lr):
    
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.lr = lr 

        self.model = Sequential()
        self.model.add(Dense(24,input_shape=(self.observation_dim,),activation='relu'))
        self.model.add(Dense(24,activation='relu'))
        self.model.add(Dense(self.num_actions,activation='linear'))
        self.model.compile(optimizer = Adam(learning_rate=self.lr),loss='mse')

    
    def predict(self,state):
        actions = self.model.predict(state,verbose=0)
        return actions
    
    def getModel(self):
        return self.model
    
    def save(self, fname):
        self.model.save(fname)
    
    def load(self, fname):
        self.model = load_model(fname)
    

