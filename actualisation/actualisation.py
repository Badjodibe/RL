import numpy as np
def actualisation(p,alpha, reward):
    """
    :param p:
    :param alpha:
    :param reward:
    :return:gain
    """
    identity = np.eye(p.shape[0], p.shape[1])
    b_ = np.array(identity-alpha*p)
    return np.linalg.solve(b_, reward)
