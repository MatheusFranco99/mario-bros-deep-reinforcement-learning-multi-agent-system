from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

class DQNNet:

    def __init__(self,num_actions):
    
        self.num_actions = num_actions

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(24,input_shape=(self.num_actions,),activation='relu'))
        self.model.add(Dense(24,activation='relu'))
        self.model.add(Dense(self.num_actions,activation='linear'))
        self.model.compile(optimizer = Adam(learning_rate=lr),loss='mse')
        return self.model
    
    def predict(self,state):
        actions = self.model.predict(state,verbose=0)
        return actions
    
    def getModel(self):
        return self.model
    
    def save(self, fname):
        self.model.save(fname)
    
    def load(self, fname):
        self.model = load_model(fname)
    

