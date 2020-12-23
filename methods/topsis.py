"""
script to calculate MCDA TOPSIS method rankings
"""
import numpy as np
from normalization import *
import copy

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
w = [0.25, 0.25, 0.25, 0.25]

class TOPSIS:
    def __init__(self, matrix, types, weights, precision, normalization=sum_normalization):
        """

        :param matrix: decision matrix
        :param types: types of criteria
        :param weights: weights for criteria
        :param precision: precision of floating point
        :param normalization: type of normalization for decision matrix
        """
        self.matrix = matrix
        self.types = types
        self.weights = weights
        self.precision = precision
        self.normalization = normalization

    def weights_multiply(self):
        """
        normalize the decision matrix and multiply every value in matrix by the weight for criteria
        :return: normalized weighted decision matrix
        """
        matrix = copy.deepcopy(self.normalized_matrix)
        m, n = self.matrix.shape
        for i in range(n):
            matrix[:, i] = self.normalized_matrix[:, i] * self.weights[i]
        return matrix

    def calculate_distance(self):
        """
        calculate positive and negative distances from ideal solution
        :return: two lists of positive and negative distance from ideal solution
        """
        PIS = np.max(self.weighted_matrix, axis=0)
        NIS = np.min(self.weighted_matrix, axis=0)

        DP, DN = [], []
        for m in self.weighted_matrix:
            DP.append(np.sqrt(sum(m - PIS) ** 2))
            DN.append(np.sqrt(sum(m - NIS) ** 2))

        return DP, DN

    def calculate_rankings(self):
        """
        calculate ranking based on calculated distances
        :return: ranking of alternatives
        """
        rankings = []
        for dn, dp in zip(self.DN, self.DP):
            rankings.append(round(dn / (dn + dp), self.precision))

        return np.array(rankings, dtype=float)

    def run(self):
        self.normalized_matrix = normalize(self.matrix, self.types, self.normalization)
        self.weighted_matrix = self.weights_multiply()
        self.DP, self.DN = self.calculate_distance()
        self.rankings = self.calculate_rankings()
