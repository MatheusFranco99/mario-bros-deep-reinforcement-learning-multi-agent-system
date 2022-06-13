from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

class DQNNet:

    def __init__(self,height,width,num_frames,num_actions):
    
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_actions = num_actions

    def build(self):
        model = Sequential()
        model.add(Convolution2D(16,(8,8),strides = (4,4), activation = 'relu',input_shape=(self.num_frames,self.height,self.width,1)))
        model.add(Convolution2D(32,(8,8),strides = (4,4), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(self.num_actions,activation='linear'))
        self.model = model
        return self.model

    def compile(self,lr):
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
    

