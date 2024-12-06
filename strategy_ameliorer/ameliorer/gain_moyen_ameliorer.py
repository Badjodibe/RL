from collections import defaultdict
from scipy.optimize import linprog
from scipy import linalg
import numpy as np

class Decision:
    def __init__(self, vertices, actions):

        self.V = vertices
        self.A = actions
        self.actions = defaultdict(list)
        self.rewards = defaultdict(list)
        self.transitions_probs = defaultdict(list)
        
    def add_actions_prob_reward(self, u, actions, rewards, transition_probs):

        self.actions[u].append(actions)
        self.rewards[u].append(rewards)
        self.transitions_probs[u].append(transition_probs)

    def strategy_transition_probability_reward(self, strategy):
        transition_probability = list()
        rewards = list()
        for i in range(self.V):
            transition_probability.append(self.transitions_probs[i][0][strategy[i]])
            rewards.append(self.rewards[i][0][strategy[i]])
        return np.array(transition_probability), np.array(rewards)

    def gain_moyen_ameliorer(self, old_strategy):
        end, new_strategy = self.new_strategy(old_strategy)
        while(end==-1):
           self.new_strategy(new_strategy)
        print(new_strategy)

    def new_strategy(self, old_strategy):
        end = -1
        transition_f, rewards_f = self.strategy_transition_probability_reward(old_strategy)
        reward_zeros = np.zeros_like(rewards_f)
        transition_zeros = np.zeros_like(transition_f)
        transitions_ones = np.ones_like(transition_f)
        trans1 = np.hstack((transition_f, transition_zeros, transition_zeros))
        trans2 = np.hstack((transitions_ones, transition_f, transition_zeros))
        trans3 = np.hstack((transition_zeros, transitions_ones, transition_f))
        a_eq = np.vstack((trans1, trans2, trans3))
        b_eq = np.vstack((reward_zeros, rewards_f, reward_zeros))
        x = linalg.solve(a_eq, b_eq.flatten())
        phi_f= x[:self.V]
        a_i_f = defaultdict(list)
        new_strategy = list()
        for i in range(self.V):
            a = np.array(np.array(self.transitions_probs[i][0])*phi_f)
            b = a > phi_f[i]
            for j in range(len(b)):
                if b[i]:
                    a_i_f[i].append(j)
        if not a_i_f:
            end = 1
        else:
            for i in range(self.V):
                if not a_i_f[i]:
                    new_strategy[i] = old_strategy[i]
                else:
                    new_strategy[i] = np.choose(a_i_f[i])
        return end, new_strategy

    def value_iteration(self, v_alpha_init,l, epsilon):
        find = False
        v_temps = np.zeros(self.A)
        v = np.zeros(self.V)
        strategy = np.zeros(self.V)
        while(find==False):
            for i in range(self.V):
                for a in range(self.A):
                    value = self.rewards[i][0][a] + l*np.dot(np.array(self.transitions_probs[i][0][a]), v_alpha_init)
                    v_temps[a] = value
                v[i] = np.max(v_temps)
                strategy[i] = np.argmax(v_temps)
                if np.abs(v_alpha_init - v).all() < epsilon:
                    find = True
                    return strategy, v 
                else:
                    v_alpha_init = v

    def linear_programming_actualisation(self, alpha):
        c = np.ones(self.V)
        A = alpha*np.array(self.transitions_probs[0][0])
        b_ub = -np.array(self.rewards[0][0])
        for i in range(1,self.V):
            A = alpha*np.vstack((A, np.array(self.transitions_probs[i][0])))
            b_ub = -np.vstack((b_ub, np.array(self.rewards[i][0])))
        b_ub = b_ub.flatten()
        var = np.zeros((self.A, self.V))
        var[:, 0] = -1
        for j in range(1,self.V):
            var_ = np.zeros((self.A, self.V))
            var_[:, j] = -1
            var = np.vstack((var, var_))
        res = linprog(c=c, A_ub=A, b_ub=b_ub, method = "simplex")
        strategy = np.zeros(self.A)
        for i in range(self.V):
            probs = np.array(self.transitions_probs[i][0])
            rew = np.array(self.rewards[i][0])
            temps = rew + alpha*np.dot(probs, res.x)
            strategy[i] = np.argmax(temps)
        return strategy
        
            
if __name__=="__main__":
    decision = Decision(3,3)
    decision.add_actions_prob_reward(0,
                                     [0,1,2],
                                     [1,2,3],
                                     [
                                            [1, 0, 0],
                                            [0,1,0],
                                         [0,0,1]
                                    ])
    decision.add_actions_prob_reward(1,
                                     [0,1,2],
                                     [6,4,5],
                                     [
                                         [1,0,0],
                                         [0,1,0],
                                         [0,0,1]
                                     ])
    decision.add_actions_prob_reward(2,
                                     [0, 1, 2],
                                     [8,9, 7],
                                     [
                                         [1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]
                                     ])
    print(decision.linear_programming_actualisation(0.5))
    