"""
script to calculate MCDA COMET method rankings
"""
import numpy as np
import copy


def tfn(x, a, m, b):
    if (x < a) or (x > b):
        return 0
    elif (a <= x) and (x < m):
        return (x-a)/(m-a)
    elif (m < x) and (x <= b):
        return (b-x)/(b-m)
    elif x == m:
        return 1

class Point:

    def __init__(self, coords=None, preference=0, color='bo'):
        """

       :param coords: characteristic object
       :param preference: preference
       """
        if coords is None:
            coords = []
        self.coords = coords
        self.preference = preference
        self.color = color

    def show(self):
        print(self.coords)
        # print(self.preference)

def scope(point, criteria):
    """
   Create space iterator for some point. Space is a list of values to tfn function (a, m, b).
   :param point: just point
   :param criteria: list of criteria
   :return: space tfn iterator
   """
    for var, criterion in zip(point.coords, criteria):
        # ind = var.index(criterion)
        ind = criterion.index(var)
        if ind == len(criterion) - 1:
            yield criterion[ind - 1], criterion[ind], criterion[ind]
        elif ind == 0:
            #    yield criterion[ind], criterion[ind], criterion[ind - 1]
            yield criterion[ind], criterion[ind], criterion[ind + 1]
        else:
            yield criterion[ind - 1], criterion[ind], criterion[ind + 1]

def comet(criteria, tab_of_points, alternatives):
    """

   :param criteria: list of all criteria
   :param tab_of_points: list of boundary space points
   :param alternatives: list of alternatives
   :return: alternatives with calculated preference value
   """
    # product = 0
    product = []
    preference = 0
    alternatives = copy.deepcopy(alternatives)
    for alternative in alternatives:
        for point in tab_of_points:
            space_iter = scope(point, criteria)
            for space, coord in zip(space_iter, alternative.coords):
                # product += tfn(coord, space[0], space[1], space[2])
                product.append(tfn(coord, space[0], space[1], space[2]))
            preference += np.product(product) * point.preference
            product = []
        alternative.preference = preference
        preference = 0
    return alternatives


def calc_model(data, C1, C2, C3, C4, Pref_C1C2, Pref_C3C4, Pref_p1p2, R=2, show=True):
    """ function to calculate outputs for selection of swimmers """


    C1C2 = []
    for i in C1:
        for j in C2:
            C1C2.append([i, j])


    model_c1c2 = []
    for i in range(len(Pref_C1C2)):
        model_c1c2.append(Point(C1C2[i][:], Pref_C1C2[i]))

    P1 = []

    for i in range(len(data)):
        alt = comet([C1, C2], model_c1c2,
                    [Point([data[i][0], data[i][1]])])
        P1.append(round(alt[0].preference, R))

    C3C4 = []
    for i in C3:
        for j in C4:
            C3C4.append([i, j])

    model_c3c4 = []
    for i in range(len(Pref_C3C4)):
        model_c3c4.append(Point(C3C4[i][:], Pref_C3C4[i]))

    P2 = []

    for i in range(len(data)):
        alt = comet([C3, C4], model_c3c4, [Point([data[i][2], data[i][3]])])
        P2.append(round(alt[0].preference, R))

    p1p2 = [[0, 0],
            [0, 0.5],
            [0, 1],
            [0.5, 0],
            [0.5, 0.5],
            [0.5, 1],
            [1, 0],
            [1, 0.5],
            [1, 1]
            ]

    model_p1p2 = []
    for i in range(len(Pref_p1p2)):
        model_p1p2.append(Point(p1p2[i][:], Pref_p1p2[i]))

    P = []

    for i in range(len(data)):
        alt = comet([[0, 0.5, 1], [0, 0.5, 1]], model_p1p2, [Point([P1[i], P2[i]])])
        P.append(round(alt[0].preference, R))

    if show:
        print("################ Wyniki ####################")
        print(P1)
        print(P2)
        print(P)

    return P
