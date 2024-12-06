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
        while (end == -1):
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
        phi_f = x[:self.V]
        a_i_f = defaultdict(list)
        new_strategy = list()
        for i in range(self.V):
            a = np.array(np.array(self.transitions_probs[i][0]) * phi_f)
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
