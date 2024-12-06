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

    def linear_programming_actualisation(self, alpha):
        c = np.ones(self.V)
        A = alpha * np.array(self.transitions_probs[0][0])
        b_ub = -np.array(self.rewards[0][0])
        for i in range(1, self.V):
            A = alpha * np.vstack((A, np.array(self.transitions_probs[i][0])))
            b_ub = -np.vstack((b_ub, np.array(self.rewards[i][0])))
        b_ub = b_ub.flatten()
        var = np.zeros((self.A, self.V))
        var[:, 0] = -1
        for j in range(1, self.V):
            var_ = np.zeros((self.A, self.V))
            var_[:, j] = -1
            var = np.vstack((var, var_))
        res = linprog(c=c, A_ub=A, b_ub=b_ub, method="simplex")
        strategy = np.zeros(self.A)
        for i in range(self.V):
            probs = np.array(self.transitions_probs[i][0])
            rew = np.array(self.rewards[i][0])
            temps = rew + alpha * np.dot(probs, res.x)
            strategy[i] = np.argmax(temps)
        return strategy