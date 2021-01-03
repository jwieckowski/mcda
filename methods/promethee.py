"""
script to calculate MCDA PROMETHEE method rankings

example:

p = np.random.rand(MenInd.shape[1]) / 2
q = np.random.rand(MenInd.shape[1]) / 2 + 0.5

p = PROMETHEE(MenInd, types, w, 'usual', p, q, 2)
p.run()

"""

import numpy as np
import copy


class PROMETHEE:
    def __init__(self, matrix, types, weights, type, p, q, precision):
        self.matrix = matrix
        self.types = types
        self.weights = weights
        self.type = type
        self.p = p
        self.q = q
        self.precision = precision

        functions = {
            'usual': self.usual_type,
            'ushape': self.ushape_type,
            'vshape': self.vshape_type,
            'level': self.level_type,
            'vshape2': self.vshape2_type
        }
        self.pref = functions[self.type]

    def usual_type(self, d, number=None):
        if d <= 0:
            return 0
        return 1

    def ushape_type(self, d, number):
        if d <= self.q[number]:
            return 0
        return 1

    def vshape_type(self, d, number):
        if d <= 0:
            return 0
        elif d <= self.p[number]:
            return d/self.p[number]
        return 1

    def level_type(self, d, number):
        if d <= self.q[number]:
            return 0
        elif d <= self.p[number]:
            return 1/2
        return 1

    def vshape2_type(self, d, number):
        if d <= self.q[number]:
            return 0
        elif d <= self.p[number]:
            return (d-self.q[number])/(self.p[number]-self.q[number])
        return 1

    def split_criteria_matrix(self):
        row, col = self.matrix.shape
        self.criteria_matrix = []

        for c in range(col):
            m = np.zeros((row, row))
            for r in range(row):
                if self.types[c] == 'cost':
                    m[:, r] = self.matrix[r, c] - self.matrix[:, c]
                else:
                    m[:, r] = self.matrix[:, c] - self.matrix[r, c]
            self.criteria_matrix.append(m)

    def do_binary_matrix(self):
        self.binary_matrix = []
        for index, m in enumerate(self.criteria_matrix):
            row, col = m.shape
            b = np.zeros((row, row))
            for r in range(row):
                for c in range(row):
                    b[r, c] = self.pref(m[r, c], index)
            self.binary_matrix.append(b)

    def calculate_pi(self):
        self.pi_matrix = copy.deepcopy(self.binary_matrix)
        self.pi_matrix = sum([np.array(b * self.weights[index]) for index, b in enumerate(self.binary_matrix)])

    def calculate_fi(self):
        row, col = self.pi_matrix.shape
        self.FP = np.sum(self.pi_matrix, axis=1) / (row-1)
        self.FM = np.sum(self.pi_matrix, axis=0) / (row-1)
        self.F = self.FP - self.FM

    def run(self):
        self.split_criteria_matrix()
        self.do_binary_matrix()
        self.calculate_pi()
        self.calculate_fi()

