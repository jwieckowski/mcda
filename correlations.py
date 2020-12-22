import numpy as np
from itertools import permutations

def _cov(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: covariance with normalized values (bias = True)
    """
    return np.cov(x, y, bias=True)[0][1]

def spearman(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: spearman correlations
    """
    return (_cov(x, y)) / (np.std(x) * np.std(y))


def pearson(x, y):
    """

    :param x: preferential ranking 1
    :param y: preferential ranking 2
    :return: pearson correlations
    """
    return (_cov(x, y)) / (np.std(x) * np.std(y))


def weighted_spearman(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: weighted spearman correlations
    """

    N = len(x)
    n = 6 * np.sum((x-y)**2 * ((N - x + 1) + (N - y + 1)))
    d = N**4 + N**3 - N**2 - N
    return 1 - (n/d)


def ws_rank(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: ws correlations
    """

    N = len(x)
    n = np.fabs(x - y)
    d = np.max((np.fabs(1 - x), np.fabs(N - x)), axis=0)
    return 1 - np.sum(2.0**(-1.0 * x) * n/d)


def kendall_tau(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: kendall tau correlations
    """

    n = len(x)
    res = 0
    for j in range(n):
        for i in range(j):
            res += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return 2/(n*(n-1)) * res


def goodman_kruskal_gamma(x, y):
    """

    :param x: positional ranking 1
    :param y: positional ranking 2
    :return: goodman kruskal correlations
    """

    num = 0
    den = 0
    for i, j in permutations(range(len(x)), 2):
        x_dir = x[i] - x[j]
        y_dir = y[i] - y[j]
        sign = np.sign(x_dir * y_dir)
        num += sign
        if sign:
            den += 1
    return num / float(den)
