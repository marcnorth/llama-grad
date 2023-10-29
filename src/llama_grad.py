import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from gradient_calculator.gradient_calculator import GradientCalculator
from token_with_gradients import TokenWithGradients
from typing import List
from visualization.html_visualizer import HtmlVisualizer


class LlamaGrad:
    """
    Keeps track af gradients through a full generation.
    Represents a single generation of sequence of text
    This class is stateful. The initial prompt is passed to the constructor.
    Calling next_token multiple times will append the preceding output to the input for generating the next token.
    i.e. if i is the initial input, calling next_token once will use i to generate the first output token, o0. Calling next_token again will use concat(i, o0) as input to generate the second output, o1.
    """

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
        self.input_encodings = tokenizer.batch_encode_plus([prompt], return_tensors="pt", add_special_tokens=False, padding=True)
        self.output_tokens_with_gradients = TokenWithGradients()
        self.gradient_calculator = gradient_calculator

    def generate(self, stop_tokens: List[str] = None, max_output_length: int = None) -> TokenWithGradients:
        """
        Generates output tokens using input calculates gradients for each output token until max_length or  EOS is reached (Maybe other settings?)
        :param stop_tokens: list of tokens to stop generating at. Defaults to self.tokenizer.eos_token
        :param max_output_length: maximum number of generated output tokens
        :return:
        """
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token]
        stop_token_ids = [self.tokenizer.encode(stop_token) for stop_token in stop_tokens]
        # Loop through next_token until max_length or until we reach stop token
        target_token_count = self.output_tokens_with_gradients.token_ids.shape[0] + max_output_length if max_output_length is not None else None
        self.next_token()
        while self.output_tokens_with_gradients.token_ids[-1].item() not in stop_token_ids and (target_token_count is None or self.output_tokens_with_gradients.token_ids.shape[0] < target_token_count):
            self.next_token()
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

        # TODO: Needs to be per-model
        input_embeddings = self.model.transformer.wte(input_ids.unsqueeze(0))
        #input_embeddings.requires_grad = True

        # Pass to gradient calculation strategy (e.g. SmoothGrad)
        token_with_gradients = self.gradient_calculator.forward(self.model, input_embeddings)

        self.output_tokens_with_gradients.append(token_with_gradients)

        return token_with_gradients

    def html_visualizer(self):
        return HtmlVisualizer(
            self.tokenizer,
            self.prompt,
            self.output_tokens_with_gradients
        )