from typing import Self
from torch import Tensor
import torch


class TokenWithGradients:
    def __init__(self):
        # 1-D Tensor of predicted token_ids of size (output_sequence_length)
        self.token_ids: Tensor = torch.zeros([0], dtype=torch.int)
        # 2-D Tensor of gradients of size (output_sequence_length, input_size + self.token_ids.shape[0] - 1)
        self.gradients: Tensor = None

    def append(self, other: Self):
        if self.gradients is None:
            self.token_ids = other.token_ids.clone()
            self.gradients = other.gradients.clone()
        else:
            zeros = torch.zeros((self.gradients.shape[0], other.gradients.shape[1] - self.gradients.shape[1]))
            self.gradients = torch.cat([self.gradients, zeros], 1)
            self.token_ids = torch.cat((self.token_ids, other.token_ids), 0)
            self.gradients = torch.cat((self.gradients, other.gradients), 0)
