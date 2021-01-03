"""
script to calculate MCDA TOPSIS method rankings

example:
matrix = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
types = ['cost', cost', 'profit']
weights = [0.2, 0.45, 0.35]
precision = 2
topsis = TOPSIS(matrix, types, weights, precision)
topsis.run()

"""
import numpy as np
from normalization import *
import copy


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

