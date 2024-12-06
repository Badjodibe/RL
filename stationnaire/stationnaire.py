import numpy as np


def find_stationary_matrix(P, recurrent_classes, transient_states):
    """
    Calculer la matrice stationnaire à partir de la matrice de transition P,
    des classes récurrentes, et des états transitoires.

    :param P: Matrice de transition (numpy array)
    :param recurrent_classes: Liste des classes récurrentes (liste de listes d'états)
    :param transient_states: Liste des états transitoires
    :return: Matrice stationnaire P*
    """
    n = P.shape[0]
    P_star = np.zeros_like(P)

    # Étape 1: Calculer les vecteurs stationnaires pour chaque classe récurrente
    for E_k in recurrent_classes:
        # Isoler la sous-matrice pour les états récurrents
        P_Ek = P[np.ix_(E_k, E_k)]

        # Résoudre l'équation pi * P_Ek = pi avec sum(pi) = 1
        A = np.vstack([P_Ek.T - np.eye(len(E_k)), np.ones(len(E_k))])
        b = np.zeros(len(E_k) + 1)
        b[-1] = 1  # Contrainte que la somme des éléments de pi doit être égale à 1

        # Résoudre le système d'équations linéaires
        pi_Ek = np.linalg.lstsq(A, b, rcond=None)[0]

        # Stocker le vecteur stationnaire dans la matrice P*
        for i, state in enumerate(E_k):
            P_star[state, E_k] = pi_Ek[i]

    # Étape 2: Calculer les composantes pour les états transitoires
    for i in transient_states:
        for k, E_k in enumerate(recurrent_classes):
            P_Ek = P[np.ix_(E_k, E_k)]

            # Calculer a_ik
            A = np.eye(len(transient_states)) - P[np.ix_(transient_states, transient_states)]
            b = P[i, E_k]
            a_ik = np.linalg.lstsq(A, b, rcond=None)[0]

            # Mettre à jour les éléments correspondants dans P*
            for j, state in enumerate(E_k):
                P_star[i, state] = a_ik[j] * P_star[state, state]

    return P_star


