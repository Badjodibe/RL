import numpy as np
import random

class QLearnig:
    def __init__(self, goal, strat, obstacle,actions, grid, gamma, alpha, epsilon, nEpoch):
        self.goal = goal
        self.start = strat
        self.obstacle = obstacle
        self.actions = actions
        self.grid = grid
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.nEpoch = nEpoch
        self.Qtable = np.zeros((self.grid.shape[0], self.grid.shape[1], len(actions)))
        
    def qleraning(self, init):
        """
            Q-Learning algorithme
        """
        i = 0
        while(i<self.nEpoch):
            action = self.state_choice(init)
            state, reward = self.perform_action(init, action)
            if np.array_equal(init, state):
                continue
            self.updateQ(init, action, state, reward)
            init =  state
        pass
    def updateQ(self, etat, action, etat_, reward):
        q = np.max(self.Qtable[etat_[0]][etat_[1]])
        self.Qtable[etat[0]][etat[1]][action] = self.Qtable[etat[0],etat[1]][action] + self.alpha*(reward  + self.gamma*q - self.Qtable[etat[0],etat[1]][action])
    def perform_action(self, etat, action):
        """
            This function return the reward and the next state
        """
        new_etat = np.zeros(2)
        reward = -100
        if action==0:
            # For up
            new_etat[0] = etat[0] - 1
            
        elif action==1:
            # for down
            new_etat[0] = etat[0] + 1
            
        elif action==2:
            # for left
            new_etat[1] = etat[1] - 1
            
        else:
            # for right
            new_etat[1] = etat[1] + 1
        new_etat[0] = int(new_etat[0])
        new_etat[1] = int(new_etat[1])
        if new_etat[0]<0 or new_etat[0]>3 or new_etat[1]<0 or new_etat[1]>3:
            new_etat = etat
            reward = 0
        else:
            reward = self.grid[new_etat[0]][new_etat[1]]
        
        return new_etat, reward
    def state_choice(self, etat):

        if random.uniform(0,1)<self.epsilon:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.Qtable[etat[0]][etat[1]])
        
if __name__=="__main__":
    # goal, strat, obstacle,actions, grid, gamma, alpha, epsilon, nEpoch
    
    grid  = QLearnig(
        goal=np.array([3,3]),
        strat=np.array([0,0]),
        obstacle=np.array([1,1]),
        actions=np.array([0,1,2,3]),
        grid=np.array([
            [0,0,0,0],
            [0,-1,0,0],
            [0,0,0,0],
            [0,0,0,4]
        ]),
        gamma=0.9,
        alpha=0.1,
        epsilon=0.1,
        nEpoch=1000
        )
    grid.qleraning(np.array([0,0]))
    
    


  