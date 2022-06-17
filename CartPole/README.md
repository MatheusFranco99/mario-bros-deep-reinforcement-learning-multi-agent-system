# AASMA Project Group 07
Project of Multi Agent System based on Gym CartPole environment and Reinforcement Learning algorithms

---
## Random Agent

To run the Random agent, go to the RandomAgent folder

```
cd RandomAgent
```

and run

```
python3 RandomAgent.py
```

---
## Q-learning Agent

To run the Q-learning agent, go to the QLearningAgent folder

```
cd QLearningAgent
```

and run

```
python3 QLearning.py
```


---
## DQN Agent

To run the DQN agent, go to the DQNAgent folder

```
cd DQNAgent
```

and run

```
python3 main.py -tr -f FILENAME -n TIMESTEPS -e EPISODES [-r]
```

to train an agent with a maximum of TIMESTEPS actions or EPISODES episodes and save it in file FILENAME.  
Or run 
```
python3 main.py -te -f FILENAME
```

to test an agent saved in file FILENAME. 

---
## PPO Agent

To run the PPO agent, go to the PPOAgent folder

```
cd PPOAgent
```

and run

```
python3 ppo_stable_baselines3.py
```
