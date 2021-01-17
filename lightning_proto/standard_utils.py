from collections import namedtuple
from itertools import product


def flatten_list(parent, recursion=1):
    for index in range(recursion):
        parent = [item for sub in parent for item in sub]
    return parent


def copy_attrs(obj_from, obj_to, attr_names):
    for attr_name in attr_names:
        if hasattr(obj_from, attr_name):
            attr = getattr(obj_from, attr_name)
            setattr(obj_to, attr_name, attr)


class Combinations():
    @staticmethod
    def get_combinations(params):
        combination = namedtuple('combination', params.keys())
        combinations = []
        for permutation in product(*params.values()):
            combinations.append(combination(*permutation))
        return combinations
