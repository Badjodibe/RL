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


    def value_iteration(self, v_alpha_init, l, epsilon):
        find = False
        v_temps = np.zeros(self.A)
        v = np.zeros(self.V)
        strategy = np.zeros(self.V)
        while not find:
            for i in range(self.V):
                for a in range(self.A):
                    value = self.rewards[i][0][a] + l * np.dot(np.array(self.transitions_probs[i][0][a]), v_alpha_init)
                    v_temps[a] = value
                v[i] = np.max(v_temps)
                strategy[i] = np.argmax(v_temps)
                if np.abs(v_alpha_init - v).all() < epsilon:
                    find = True
                    return strategy, v
                else:
                    v_alpha_init = v