"""
script to calculate MCDA Interval TOPSIS method rankings

example:
matrix = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
types = ['cost', cost', 'profit']
weights = [0.2, 0.45, 0.35]
range = [0.95, 1.05]
precision = 2
it = INTERVAL_TOPSIS(matrix, types, weights, range, precision)
it.run()

"""
import numpy as np
import copy
import random

class INTERVAL_TOPSIS:
    def __init__(self, matrix, types, weights, range, precision):
        """

        :param matrix: decision matrix
        :param types: types of criteria
        :param weights: weights for criteria
        :param range: list with 2 elements for interval range
        :param precision: precision of floating point
        """
        self.matrix = matrix
        self.types = types
        self.weights = weights
        self.range = range
        self.precision = precision

    def create_interval(self):
        """
        creates a matrix with interval values based on given ranges
        :return: interval matrix
        """
        R = [random.randrange(self.range[0]*100, self.range[1]*100, 1)/100 for i in range(5)]
        row, col = self.matrix.shape
        self.interval_matrix = [[[] for c in range(col)] for r in range(row)]
        for r in range(row):
            for c in range(col):
                intervals = sorted([round(self.matrix[r][c]*weight, self.precision) for weight in R])
                self.interval_matrix[r][c] = [intervals[0], intervals[-1]]
        self.interval_matrix = np.array(self.interval_matrix)

    def normalize_matrix(self):
        """
        normalize interval matrix
        :return: normalized interval matrix
        """
        row, col, l = self.interval_matrix.shape
        self.normalized_matrix = copy.deepcopy(self.interval_matrix)
        for c in range(col):
            d = np.sqrt(np.sum(np.array(self.interval_matrix[:, c]**2).flatten()))
            self.normalized_matrix[:, c] = self.interval_matrix[:, c] / d

    def weights_multiply(self):
        """
        multiply every interval value in matrix by the weight for criteria
        :return: weighted normalized decision matrix
        """
        self.weighted_normalized_matrix = copy.deepcopy(self.normalized_matrix)
        row, col, l = self.normalized_matrix.shape
        for c in range(col):
            self.weighted_normalized_matrix[:, c] = self.normalized_matrix[:, c] * self.weights[c]


    def calculate_distance(self):
        """
        calculate positive and negative distances from ideal solution
        :return: two lists of positive and negative distance from ideal solution
        """

        self.PIS = []
        self.NIS = []

        for t in self.types:
            if t == 'profit':
                self.PIS.append(np.max(self.weighted_normalized_matrix[:, types.index(t)]))
                self.NIS.append(np.min(self.weighted_normalized_matrix[:, types.index(t)]))
            else:
                self.PIS.append(np.min(self.weighted_normalized_matrix[:, types.index(t)]))
                self.NIS.append(np.max(self.weighted_normalized_matrix[:, types.index(t)]))

        self.DP, self.DN = [], []
        row, col, l = self.weighted_normalized_matrix.shape
        for r in range(row):
            self.DP.append(np.sqrt(np.sum((self.weighted_normalized_matrix[r, :][:, 0] - self.PIS)**2) + np.sum((self.weighted_normalized_matrix[r, :][:, 1] - self.PIS)**2)))
            self.DN.append(np.sqrt(np.sum((self.weighted_normalized_matrix[r, :][:, 0] - self.NIS)**2) + np.sum((self.weighted_normalized_matrix[r, :][:, 1] - self.NIS)**2)))

    def calculate_rankings(self):
        """
        calculate ranking based on calculated distances
        :return: ranking of alternatives
        """
        self.rankings = []
        for dn, dp in zip(self.DN, self.DP):
            self.rankings.append(np.round(dn / (dn + dp), self.precision))

        self.rankings = np.array(self.rankings, dtype=float)

    def run(self):
        self.create_interval()
        self.normalize_matrix()
        self.weights_multiply()
        self.calculate_distance()
        self.calculate_rankings()
