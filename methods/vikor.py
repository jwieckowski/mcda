"""
script to calculate MCDA VIKOR method rankings
"""
import numpy as np

MenInd = np.array([
    [15, 32, 782, 1],
    [14, 32, 780, 1],
    [13, 32, 769, 2],
    [12, 32, 769, 2],
    [11, 32, 765, 2],
    [10, 32, 735, 2],
    [9,  31, 707, 2],
    [8,  30, 686, 2],
    [7,  29, 662, 3],
    [6,  27, 614, 5],
    [5,  25, 580, 6]
])
types = ['cost', 'profit', 'profit', 'cost']

def get_characteristic_values(matrix, types):
    """

    :param matrix:
    :param types:
    :return:
    """
    m, n = matrix.shape
    fP = []
    fM = []
    for i in range(n):
        if types[i] == 'profit':
            fP.append(max(matrix[:, i]))
            fM.append(min(matrix[:, i]))
        elif types[i] == 'cost':
            fP.append(min(matrix[:, i]))
            fM.append(max(matrix[:, i]))
        else:
            print('Wrong type of criteria ' + str(i+1))
    return fP, fM

def get_difference(matrix, values):

    results = []
    m, n = matrix.shape
    for i in range(n):
        results.append(values[i] - matrix[:, i])
    return np.array(results).T

def get_weighted_value(matrix, fP, fM, weights):

    results = []
    m, n = matrix.shape
    for i in range(n):
        f = fP[i] - fM[i]
        if f == 0:
            f = 10 ** -10
        results.append(weights[i] * (matrix[:, i]) / f)
    return np.array(results).T

def calculate_rankings(matrix, Rd):
    m, n = matrix.shape
    S = np.round(matrix.sum(axis=1), Rd)
    R = np.array([round(max(matrix[i, :]), Rd) for i in range(m)])

    v = 0.5
    bestS = max(S)
    worstS = min(S)
    bestR = max(R)
    worstR = min(R)
    diffS = bestS - worstS
    diffR = bestR - worstR

    Q = []
    for s, r in zip(S, R):
        qj = v * (s - worstS) / diffS + (1 - v) * (r - worstR) / diffR
        Q.append(round(qj, Rd))

    return S, R, Q

def vikor(matrix, types, weights, Rd = 2):
    """

    :param matrix:
    :param types:
    :param weights:
    :param R:
    :return:
    """
    fP, fM = get_characteristic_values(matrix, types)
    diff = get_difference(matrix, fP)
    weighted = get_weighted_value(diff, fP, fM, weights)
    S, R, Q = calculate_rankings(weighted, Rd)
    return S, R, Q

