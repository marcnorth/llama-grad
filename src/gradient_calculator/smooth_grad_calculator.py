import torch
from torch import Tensor
from transformers import PreTrainedModel
from gradient_calculator.gradient_calculator import GradientCalculator
from token_with_gradients import TokenWithGradients


class SmoothGradCalculator(GradientCalculator):
    def __init__(self, n_samples: int = 20, sigma: float = 0.005):
        self.n_samples: int = n_samples
        self.sigma: float = sigma

    def forward(self, model: PreTrainedModel, input_embeddings: Tensor) -> TokenWithGradients:
        input_embeddings.retain_grad()

        # Get the output without noise
        output = model.forward(inputs_embeds=input_embeddings)

        cumulative_gradients = torch.zeros_like(input_embeddings).norm(dim=-1)

        for _ in range(self.n_samples):
            sample_gradients = self.grad_sample(model, input_embeddings)
            cumulative_gradients += sample_gradients

        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output.logits.argmax(dim=2)[0, -1:]
        token_with_gradients.gradients = cumulative_gradients / self.n_samples

        return token_with_gradients

    def grad_sample(self, model: PreTrainedModel, input_embeddings: Tensor) -> Tensor:
        """
        A single step of smooth grad
        :param model:
        :param input_embeddings:
        :return: The magnitudes of the gradients for each input token
        """
        input_embeddings_with_noise = input_embeddings + torch.randn_like(input_embeddings) * self.sigma

        if input_embeddings_with_noise.grad is not None:
            input_embeddings_with_noise.grad.zero_()

        output = model.forward(inputs_embeds=input_embeddings_with_noise)

        # output is batch, but batch size is always one
        current_logits = output.logits[0].sum()
        current_logits.backward(retain_graph=True)

        # Magnitude of gradient for w.r.t each input token
        return torch.norm(input_embeddings.grad[0].clone(), dim=-1).unsqueeze(0)
