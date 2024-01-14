import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase, LlamaForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from typing import List, Callable, Type

from .input_importance_calculator import InputImportanceCalculator
from .token_with_gradients import TokenWithGradients
from .gradient_calculator.gradient_calculator import GradientCalculator
from .visualization.html_visualizer import HtmlVisualizer


class LlamaGrad:
    """
    Keeps track af gradients through a full generation.
    Represents a single generation of sequence of text
    This class is stateful. The initial prompt is passed to the constructor.
    Calling next_token multiple times will append the preceding output to the input for generating the next token.
    i.e. if i is the initial input, calling next_token once will use i to generate the first output token, o0. Calling next_token again will use concat(i, o0) as input to generate the second output, o1.
    """

    embedding_functions: dict[PreTrainedModel, Callable[[PreTrainedModel, Tensor], Tensor]] = {
        GPT2LMHeadModel: lambda model, token_ids: model.transformer.wte(token_ids),
        LlamaForCausalLM: lambda model, token_ids: model.model.embed_tokens(token_ids)
    }

    def __init__(
            self,
            model: PreTrainedModel,
            prompt: str,
            tokenizer: PreTrainedTokenizerBase,
            gradient_calculator: GradientCalculator
    ):
        """
        :param model: The LLM to use for generation
        :param prompt: Input prompt
        :param tokenizer: Tokenizer to encode prompt
        :param gradient_calculator: Strategy for calculating input gradients
        """
        self.model = model
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.input_encodings = tokenizer.batch_encode_plus([prompt], return_tensors="pt", add_special_tokens=False, padding=False).to(self.model.device)
        self.output_tokens_with_gradients = TokenWithGradients(self.model.device)
        self.gradient_calculator = gradient_calculator

    def generate(self, stop_tokens: List[str] = None, max_output_length: int = None, callback: Callable[[TokenWithGradients], None] = None) -> TokenWithGradients:
        """
        Generates output tokens using input calculates gradients for each output token until max_length or  EOS is reached (Maybe other settings?)
        :param stop_tokens: list of tokens to stop generating at. Defaults to self.tokenizer.eos_token
        :param max_output_length: maximum number of generated output tokens
        :return:
        """
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token]
        stop_token_ids = [self.tokenizer.encode(stop_token, add_special_tokens=False)[0] for stop_token in stop_tokens]
        # Loop through next_token until max_length or until we reach stop token
        target_token_count = self.output_tokens_with_gradients.token_ids.shape[0] + max_output_length if max_output_length is not None else None
        self.next_token()
        while self.output_tokens_with_gradients.token_ids[-1].item() not in stop_token_ids and (target_token_count is None or self.output_tokens_with_gradients.token_ids.shape[0] < target_token_count):
            self.next_token()
            if callback is not None:
                callback(self.output_tokens_with_gradients)
        # Copy just the newly generated tokes/gradients
        return_token_with_gradients = TokenWithGradients()
        return_token_with_gradients.token_ids = self.output_tokens_with_gradients.token_ids[-max_output_length:] if max_output_length else self.output_tokens_with_gradients.token_ids.clone()
        return_token_with_gradients.gradients = self.output_tokens_with_gradients.gradients[-max_output_length:] if max_output_length else self.output_tokens_with_gradients.gradients.clone()
        return return_token_with_gradients

    def next_token(self) -> TokenWithGradients:
        """
        Generates next output token and calculates gradients
        :return:
        """

        # Append outputs to inputs to create full input
        input_ids = torch.cat((self.input_encodings["input_ids"][0], self.output_tokens_with_gradients.token_ids), dim=0) if self.output_tokens_with_gradients.token_ids is not None else self.input_encodings["input_ids"][0]

        input_embeddings = self.embed_token_ids(input_ids.unsqueeze(0))
        #input_embeddings.requires_grad = True

        # Pass to gradient calculation strategy (e.g. SmoothGrad)
        token_with_gradients = self.gradient_calculator.forward(self.model, input_embeddings)

        self.output_tokens_with_gradients.append(token_with_gradients)

        return token_with_gradients

    def embed_token_ids(self, token_ids: Tensor) -> Tensor:
        if type(self.model) not in self.embedding_functions:
            raise ModelEmbeddingNotSupported(f"Embedding model {type(self.model)} is not supported. To add an embedding function, call LlamaGrad.register_embedding_function()")
        return self.embedding_functions[type(self.model)](self.model, token_ids)

    def generated_output(self) -> str:
        """
        Returns the generated output text
        :return:
        """
        return self.tokenizer.decode(self.output_tokens_with_gradients.token_ids)

    def html_visualizer(self) -> HtmlVisualizer:
        return HtmlVisualizer(self.input_importance_calculator())

    def input_importance_calculator(self) -> InputImportanceCalculator:
        return InputImportanceCalculator(
            self.tokenizer,
            self.prompt,
            self.output_tokens_with_gradients
        )

    @staticmethod
    def register_embedding_function(model_class: Type[PreTrainedModel], embedding_function: Callable[[PreTrainedModel, Tensor], Tensor]):
        """
        Registers an embedding function. Since embedding in PreTrainedModels is not consistent, we need
        to know how to embed token_ids for the used model. E.g.
        LlamaGrad.register_embedding_function(GPT2LMHeadModel), lambda model, token_ids: model.transformer.wte(token_ids))
        :param model_class:
        :param embedding_function:
        :return:
        """
        LlamaGrad.embedding_functions[model_class] = embedding_function

class ModelEmbeddingNotSupported(Exception):
    pass
