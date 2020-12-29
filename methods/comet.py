"""
script to calculate MCDA COMET method rankings

schema of performance:

- create comet object
- add criteria as a list of combined criteria considered in one submodel
- add preferences as a list for designed submodel
- run the comet method
- results contain preferences for alternatives for submodel

in case of having more than one line of depth for submodels
- follow instructions above
- add criteria as a list for combined submodels
- add preferences as a list for combined submodels
- add columns of data from received comet results (property results from COMET object)
- run the comet method

example of comet application with two lines of depth of models:
c = COMET(MenInd, 2)
c.add_criteria([C1, C2])
c.add_criteria([C3, C4])
c.add_preference(Pref_C1C2)
c.add_preference(Pref_C3C4)
c.run()

c.add_criteria([[0, 0.5, 1], [0, 0.5, 1]])
c.add_preference(Pref_p1p2)
c.add_data(c.results[0])
c.add_data(c.results[1])
c.run()

"""

import numpy as np
import copy
from itertools import product, chain

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

class COMET:
    def __init__(self, data, precision=2):
        """

        :param data: data from decision matrix
        :param precision: precision of floating point
        """
        self.data = data
        self.criteria = []
        self.results = []
        self.preferences = []
        self.precision = precision

    def add_data(self, column):
        """

        :param column: list with data with expected length same as amount of alternatives
        :return: append new column of data to stored data
        """
        col = np.array([[i] for i in column])
        self.data = np.append(self.data, col, axis=1)

    def add_criteria(self, criteria):
        """

        :param criteria: list with criteria
        :return: append list to stored criteria
        """
        self.criteria.append(criteria)

    def add_preference(self, preference):
        """

        :param preference: lit with preferences for criteria
        :return: append list to stored preferences
        """
        self.preferences.append(preference)

    def create_scope(self, point, criteria):
        """
        Create space iterator for some point. Space is a list of values to tfn function (a, m, b).
        :param point: just point
        :param criteria: list of criteria
        :return: space tfn iterator
        """
        for var, criterion in zip(point.coords, criteria):
            ind = criterion.index(var)
            if ind == len(criterion) - 1:
                yield criterion[ind - 1], criterion[ind], criterion[ind]
            elif ind == 0:
                yield criterion[ind], criterion[ind], criterion[ind + 1]
            else:
                yield criterion[ind - 1], criterion[ind], criterion[ind + 1]

    def calculate_preference(self, criteria, points, alternatives):
        """

        :param criteria: criteria with characteristic values
        :param points: list of points of preferences created with Point class
        :param alternatives: list of points of alternatives created with Point class
        :return:
        """
        product = []
        preference = 0
        alternatives = copy.deepcopy(alternatives)
        for alternative in alternatives:
            for point in points:
                space_iter = self.create_scope(point, criteria)
                for space, coord in zip(space_iter, alternative.coords):
                    product.append(tfn(coord, space[0], space[1], space[2]))
                preference += np.product(product) * point.preference
                product = []
            alternative.preference = preference
            preference = 0
        return alternatives

    def run(self):
        """
        application of comet method calculations
        :return: lists of preferences for comet submodels and models for alternatives
        """

        if len(self.criteria) != len(self.preferences):
            print('Size of criteria and preferences does not match')
            return

        self.results = []

        for criteria, preference in zip(self.criteria, self.preferences):
            perm_criteria = list(product(*criteria))
            model = [Point(list(perm_criteria[p][:]), preference[p]) for p in range(len(preference))]

            res = []

            index = self.criteria.index(criteria)
            amount = len(list(chain(*self.criteria[0:index])))

            for i in range(len(self.data)):
                alt = self.calculate_preference(criteria, model, [Point([self.data[i][amount + j] for j in range(len(criteria))])])
                res.append(round(alt[0].preference, self.precision))
            self.results.append(res)


