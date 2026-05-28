import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,col_num,row_num):
        self.col_num = col_num
        self.row_num = row_num
        self.x = 0
        self.y = row_num-1

    def step(self,action):
        change = [[1,0],[-1,0],[0,1],[0,-1]]
        self.x = min(self.col_num-1,max(0,self.x+change[action][0]))
        self.y = min(self.row_num-1,max(0,self.y+change[action][1]))
        next_state = self.y*self.col_num+self.x
        reward = -1
        done = False

        if self.y ==self.row_num-1 and self.x > 0:
            done = True
            if self.x != self.col_num-1:
                reward = -50

        return next_state,reward,done

    def reset(self):
        self.x = 0
        self.y = self.row_num-1
        return self.y*self.col_num+self.x

    class Sarsa:
        def __init__(self,col_num,row_num,epsilon,alpha,gamma,n_action=4):
            self.Q_table = np.zeros([col_num*row_num,n_action])
            self.epsilon = epsilon
            self.alpha = alpha
            self.gamma = gamma
            self.n_action = n_action

        def take_action(self,state):
            random_a = np.random.random()
            if random_a <= self.epsilon:
                action = np.random.randint(self.n_action)
            else:
                action = np.argmax(self.Q_table[state])
            return action

        def best_action(self,state):
            pass

        def update(self,s0,a0,r,s1,a1):
            self.Q_table[(s0,a0)]= self.Q_table[(s0,a0)]+self.alpha*(r+self.gamma*self.Q_table[(s1,a1)]-self.Q_table[(s0,a0)])

