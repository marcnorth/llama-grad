from abc import ABC, abstractmethod
from torch import Tensor
from transformers import PreTrainedModel
from token_with_gradients import TokenWithGradients


class GradientCalculator(ABC):
    @abstractmethod
    def forward(self, model: PreTrainedModel, input_embeddings: Tensor) -> TokenWithGradients:
        pass
