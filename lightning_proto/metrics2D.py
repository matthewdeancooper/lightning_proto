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

# from typing import Optional, Sequence, Union

import numpy as np
# import rising
# import rising.transforms
import torch
import torch.nn.functional as F


def bce_with_logits(*args):
    return F.binary_cross_entropy_with_logits(*args)


# def bce_loss(true, logits, pos_weight=None):
#     """Computes the weighted binary cross-entropy loss.
#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, 1, H, W]. Corresponds to
#             the raw output or logits of the model.
#         pos_weight: a scalar representing the weight attributed
#             to the positive class. This is especially useful for
#             an imbalanced dataset.
#     Returns:
#         bce_loss: the weighted binary cross-entropy loss.
#     """
#     bce_loss = F.binary_cross_entropy_with_logits(
#         logits.float(),
#         true.float(),
#         pos_weight=pos_weight,
#     )
#     return bce_loss

# def dice_loss(true, logits, eps=1e-7):
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     print(true.shape)
#     print(logits.shape)

#     true = true.type(torch.bool)

#     num_classes = logits.shape[1]
#     print(num_classes)
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0, ) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return (1 - dice_loss)

# loss_dict = {"bce_with_logits": bce_with_logits, "dice_loss": dice_loss}

loss_dict = {"bce_with_logits": bce_with_logits}

# loss = loss_dict["bce_with_logits"]
# print(loss(x, y))

# loss = loss_dict["dice_loss"]
# print(loss(x, y))
