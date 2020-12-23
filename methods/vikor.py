"""
script to calculate MCDA VIKOR method rankings
"""
import numpy as np

class VIKOR:
    def __init__(self, matrix, types, weights, precision):
        """

        :param matrix: decision matrix
        :param types: types of criteria
        :param weights: weights for criteria
        :param precision: precision of floating point
        """
        self.matrix = matrix
        self.types = types
        self.weights = weights
        self.precision = precision

    def get_extreme_values(self):
        """
        calculate two lists of maximal and minimal values from row of decision matrix
        :return: lists of maximum and minimum values
        """

        m, n = self.matrix.shape
        fP, fM = [], []
        for i in range(n):
            if self.types[i] == 'profit':
                fP.append(max(self.matrix[:, i]))
                fM.append(min(self.matrix[:, i]))
            elif self.types[i] == 'cost':
                fP.append(min(self.matrix[:, i]))
                fM.append(max(self.matrix[:, i]))
            else:
                print('Wrong type of criteria ' + str(i + 1))
        return fP, fM

    def get_difference(self):
        """
        calculate difference between list with maximal value and particular value in matrix
        :return: matrix with difference values
        """

        results = []
        m, n = self.matrix.shape
        for i in range(n):
            results.append(self.FP[i] - self.matrix[:, i])
        return np.array(results).T

    def get_weighted_matrix(self):
        """
        calculate weights list based on difference between minimal and maximal values lists
        :return: list with weighted values
        """

        results = []
        m, n = self.difference.shape
        for i in range(n):
            diff = self.FP[i] - self.FM[i]
            if diff == 0:
                diff = 10 ** -10
            results.append(self.weights[i] * (self.difference[:, i]) / diff)
        return np.array(results).T

    def calculate_rankings(self):
        """
        calculate rankings S, R, Q for calculated matrix
        :return: rankings S, R, Q
        """
        m, n = self.weighted_matrix.shape
        S = np.round(self.weighted_matrix.sum(axis=1), self.precision)
        R = np.array([round(max(self.weighted_matrix[i, :]), self.precision) for i in range(m)])

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
            Q.append(round(qj, self.precision))

        return S, R, Q

    def run(self):
        self.FP, self.FM = self.get_extreme_values()
        self.difference = self.get_difference()
        self.weighted_matrix = self.get_weighted_matrix()
        self.rankings = self.calculate_rankings()


