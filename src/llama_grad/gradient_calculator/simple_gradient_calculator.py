import torch
from torch import Tensor
from transformers import PreTrainedModel
from llama_grad.gradient_calculator.gradient_calculator import GradientCalculator
from llama_grad.token_with_gradients import TokenWithGradients


class SimpleGradientCalculator(GradientCalculator):
    def forward(self, model: PreTrainedModel, input_embeddings: Tensor) -> TokenWithGradients:
        input_embeddings.retain_grad()

        output = model.forward(inputs_embeds=input_embeddings)

        if input_embeddings.grad is not None:
            input_embeddings.grad.zero_()

        # output is batch, but batch size is always one
        model.zero_grad()
        max_logits = output.logits[0].max()
        max_logits.backward(retain_graph=True)

        # Magnitude of gradient for w.r.t each input token
        gradients = torch.norm(input_embeddings.grad[0].clone(), dim=-1).unsqueeze(0)

        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output.logits.argmax(dim=2)[0, -1:]
        token_with_gradients.gradients = gradients

        return token_with_gradients
