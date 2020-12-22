"""
script to calculate MCDA TOPSIS method rankings
"""
import numpy as np
from normalization import *

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

def normalize_matrix(matrix, types, method):
    normalized_matrix = normalize(matrix, types, method)
    return normalized_matrix

def weights_multiply(matrix, weights):
    m, n = matrix.shape
    for i in range(n):
        matrix[:, i] = matrix[:, i] * weights[i]
    return matrix

def calculate_distance(matrix):

    PIS = np.max(matrix, axis=0)
    NIS = np.min(matrix, axis=0)

    DP = []
    DN = []
    for m in matrix:
        DP.append(np.sqrt(sum(m - PIS) ** 2))
        DN.append(np.sqrt(sum(m - NIS) ** 2))

    return DP, DN

def calculate_rankings(DP, DN, R = 2):
    rankings = []
    for dm, dp in zip(DN, DP):
        rankings.append(round(dm / (dm + dp), R))

    return np.array(rankings, dtype=float)

def topsis(matrix, types, weights, R = 2, method = sum_normalization):
    """

    :param matrix:
    :param types:
    :param weights:
    :param R: fraction position to round
    :param method:
    :return:
    """

    normalized_matrix = normalize_matrix(matrix, types, method)
    weighted = weights_multiply(normalized_matrix, weights)
    DP, DN = calculate_distance(weighted)
    ranking = calculate_rankings(DP, DN, R)

    return ranking
