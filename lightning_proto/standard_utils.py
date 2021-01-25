# Copyright (C) 2020 Matthew Cooper

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import namedtuple
from itertools import product


def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)


def flatten_list(parent, recursion=1):
    for index in range(recursion):
        parent = [item for sub in parent for item in sub]
    return parent


def copy_attrs(obj_from, obj_to, attr_names):
    for attr_name in attr_names:
        if hasattr(obj_from, attr_name):
            attr = getattr(obj_from, attr_name)
            setattr(obj_to, attr_name, attr)


class Combinations:
    @staticmethod
    def get_combinations(params):
        combination = namedtuple("combination", params.keys())
        combinations = []
        for permutation in product(*params.values()):
            combinations.append(combination(*permutation))
        return combinations
