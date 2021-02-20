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

import numpy as np
import torch
import torch.nn.functional as F


def bce_with_logits(*args):
    return F.binary_cross_entropy_with_logits(*args)


loss_dict = {"bce_with_logits": bce_with_logits}

x = torch.randn(3)
y = torch.empty(3).random_(2)

loss = loss_dict["bce_with_logits"]
