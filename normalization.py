"""
script to matrix normalization
"""

from functools import reduce
import math as m
import numpy as np


def minmax_normalization(x, type):
    """

    :param x: column of matrix data
    :param type: type of normalization
    :return: min max normalized column of matrix data
    """
    if min(x) == max(x):
        return np.ones(x.shape)

    if type == 'cost':
        return (max(x) - x) / (max(x) - min(x))
    return (x - min(x)) / (max(x) - min(x))


def max_normalization(x, type):
    """

    :param x: column of matrix data
    :param type: type of normalization
    :return: max normalized column of matrix data
    """

    if type == 'cost':
        return 1 - x/max(x)
    return x / max(x)


def sum_normalization(x, type):
    """

    :param x: column of matrix data
    :param type: type of normalization
    :return: sum normalized column of matrix data
    """
    if type == 'cost':
        return (1/x) / sum(1/x)
    return x / sum(x)


def vector_normalization(x, type):
    """

    :param x: column of matrix data
    :param type: type of normalization
    :return: vector normalized column of matrix data
    """
    if type == 'cost':
        return 1 - (x / np.sqrt(sum(x ** 2)))
    return x / np.sqrt(sum(x ** 2))


def logaritmic_normalization(x, type):
    """

    :param x: column of matrix data
    :param type: type of normalization
    :return: logarithmic normalized column of matrix data
    """
    prod = reduce(lambda a, b: a*b, x)
    if type == 'cost':
        return (1 - (np.log(x) / m.log(prod))) / (len(x) - 1)
    return np.log(x) / m.log(prod)

def normalize(matrix, types, method, precision = 2):
    """

    :param matrix: decision matrix
    :param types: types of normalization for columns
    :param method: method of normalization
    :param precision: precision
    :return: normalized matrix
    """

    if matrix.shape[1] != len(types):
        print('Sizes does not match')

    normalized_matrix = matrix.astype('float')
    for i in range(len(types)):
        if type == 1:
            normalized_matrix[:, i] = np.round(method(matrix[:, i], types[i]), precision)
        else:
            normalized_matrix[:, i] = np.round(method(matrix[:, i], types[i]), precision)
    return normalized_matrix
