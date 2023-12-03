from __future__ import annotations
import json
from enum import Enum
from statistics import mean
from typing import List, Tuple, Optional, Union, TextIO
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from llama_grad import TokenWithGradients


class GroupGradientPooling(Enum):
    AVERAGE = 1
    MAX = 2


class OutputGradientPooling(Enum):
    AVERAGE = 1
    MAX = 2


class MaxGradient(Enum):
    ALL_OUTPUTS = 1  # Uses max gradient across all outputs
    SINGLE_OUTPUT = 2  # Uses max gradient of only the output html is being generated for


class InputImportanceCalculator:
    """
    Calculates the importance of each input token (or group of input tokens)
    for each output token given the gradients
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt: str, token_with_gradients: TokenWithGradients):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.input_token_ids: List[int] = tokenizer.batch_encode_plus([prompt], add_special_tokens=False)["input_ids"][0]
        self.input_token_count = len(self.input_token_ids)
        self.token_with_gradients = token_with_gradients

    def calculate_importance_for_all_outputs(
            self,
            output_gradient_pooling: OutputGradientPooling = OutputGradientPooling.AVERAGE,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = GroupGradientPooling.AVERAGE,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False
    ) -> Tuple[List[float], List[int]]:
        """
        Calculates the importance of each input token for all outputs
        :return: Tuple[List of importance scores, List of token ids (or groups)]
        """
        input_ids = self._get_input_ids_for_calculation(prompt_only=True, ignore=ignore)
        pooled_gradients = torch.mean(self.token_with_gradients.gradients[:, :len(input_ids)], dim=0) \
            if output_gradient_pooling == OutputGradientPooling.AVERAGE else \
            torch.max(self.token_with_gradients.gradients[:, :len(input_ids)], dim=0).values
        input_gradients = [gradient.item() for gradient, input_id in zip(pooled_gradients, input_ids) if input_id is not None]
        return self._calculate_grouped_importance(input_ids, input_gradients, groups, group_gradient_pooling, MaxGradient.SINGLE_OUTPUT, ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens)

    def calculate_importance_for_nth_output(
            self,
            output_token_index: int,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = GroupGradientPooling.AVERAGE,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            prompt_only: bool = False,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False
    ) -> Tuple[List[float], List[int]]:
        """
        Calculates the importance of each input token for the nth output token
        :param output_token_index: The index of the output token to calculate importance for
        :param groups: List of strings from input to group together
        :param group_gradient_pooling: How to pool the gradients of grouped input tokens
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have importance=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: Tuple[List of importance scores, List of token ids (or groups)]
        """
        input_ids = self._get_input_ids_for_calculation(output_token_index, prompt_only, ignore)
        # Only include gradients where input_ids is not None
        gradients = self.token_with_gradients.gradients[output_token_index]
        input_gradients = [gradient.item() for gradient, input_id in zip(gradients, input_ids) if input_id is not None]
        return self._calculate_grouped_importance(input_ids, input_gradients, groups, group_gradient_pooling, max_gradient, ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens)

    def _calculate_grouped_importance(
            self,
            input_ids: List[Optional[int]],
            input_gradients: List[float],
            groups: List[str],
            group_gradient_pooling: GroupGradientPooling,
            max_gradient: Union[MaxGradient, float],
            ignore_non_grouped_input_tokens: bool = False
    ) -> Tuple[List[float], List[int]]:
        """
        Groups gradients and calculates importance
        :param input_ids:
        :param input_gradients:
        :param groups:
        :param group_gradient_pooling:
        :param max_gradient:
        :return: Tuple[List of importance scores, List of token ids (or groups)]
        """
        input_ids = list(filter(lambda x: x is not None, input_ids))
        grouped_input_ids = []
        grouped_input_gradients = []
        continue_searching_from = 0  # Index to continue searching for group from
        for group in groups:
            # Look for group in input_ids
            group_token_ids = self.tokenizer.encode(group, add_special_tokens=False)
            found_index = None
            for i in range(continue_searching_from, len(input_ids) - len(group_token_ids) + 1):
                # For consistency, decode then encode the sequence we're checking against (sequence may have been encoded differently when part of a larger sequence)
                decoded_then_encoded = self.tokenizer.encode(self.tokenizer.decode(input_ids[i:i + len(group_token_ids)]), add_special_tokens=False)
                if decoded_then_encoded == group_token_ids:
                    found_index = i
                    break
            if found_index is None:
                raise ValueError(f"Group '{group}' not found")
            # Add all individual tokens before group, then add group
            if found_index > continue_searching_from:
                grouped_input_ids.extend(input_ids[continue_searching_from:found_index])
                grouped_input_gradients.extend(input_gradients[continue_searching_from:found_index] if not ignore_non_grouped_input_tokens else [0.] * (found_index - continue_searching_from))
            grouped_input_ids.append(group_token_ids)
            # Calculate group gradient using pooling strategy
            group_gradient = mean(input_gradients[found_index:found_index + len(
                group_token_ids)]) if group_gradient_pooling == GroupGradientPooling.AVERAGE else max(
                input_gradients[found_index:found_index + len(group_token_ids)])
            grouped_input_gradients.append(group_gradient)
            continue_searching_from = found_index + len(group_token_ids)
        # Add any tokens not already added
        grouped_input_ids.extend(input_ids[continue_searching_from:])
        grouped_input_gradients.extend(input_gradients[continue_searching_from:] if not ignore_non_grouped_input_tokens else [0.] * (len(input_ids) - continue_searching_from))
        max_input_gradient = (
            max(grouped_input_gradients) if max_gradient == MaxGradient.SINGLE_OUTPUT else
            torch.max(self.token_with_gradients.gradients).item() if max_gradient == MaxGradient.ALL_OUTPUTS else
            max_gradient)
        return (
            list(map(lambda g: g / max_input_gradient, grouped_input_gradients)),
            grouped_input_ids
        )

    def _get_input_ids_for_calculation(self, output_token_index: Optional[int] = None, prompt_only: bool = False, ignore: List[str] = []) -> List[Optional[int]]:
        """
        Returns the input token ids to use for calculating the nth output token
        :param output_token_index: The index of the output token to calculate importance for (None if all outputs). Must be non-None if prompt_only is False
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: input_id will be None if the token should be ignored
        """
        if prompt_only:
            # Only use the prompt input tokens and gradients
            input_ids = self.input_token_ids
        else:
            # Append the previous output tokens (0 to output_token_index-1) to the input
            input_ids = self.input_token_ids + self.token_with_gradients.token_ids[:output_token_index].tolist()
        for ignore_str in ignore:
            # If ignore_str is in input_ids, change to None
            ignore_token_ids = self.tokenizer.encode_plus(ignore_str, add_special_tokens=False)["input_ids"]
            for i in range(0, len(input_ids) - len(ignore_token_ids) + 1):
                if input_ids[i:i + len(ignore_token_ids)] == ignore_token_ids:
                    input_ids[i:i + len(ignore_token_ids)] = [None] * len(ignore_token_ids)
        return input_ids

    def save(self, file_handle: TextIO) -> None:
        file_handle.seek(0)
        # Json encode
        json_dict = {
            "tokenizer_name": self.tokenizer.name_or_path,
            "prompt": self.prompt,
            "token_with_gradients": {
                "token_ids": self.token_with_gradients.token_ids.tolist(),
                "gradients": self.token_with_gradients.gradients.tolist()
            }
        }
        file_handle.write(json.dumps(json_dict))
        file_handle.truncate()

    @staticmethod
    def load(file_handle: TextIO, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> InputImportanceCalculator:
        """
        Loads an InputImportanceCalculator from a file handle
        :param file_handle:
        :param tokenizer: If None, will use the tokenizer_name in the file to load the tokenizer using AutoTokenizer.from_pretrained()
        :return:
        """
        json_dict = json.load(file_handle)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(json_dict["tokenizer_name"])
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = Tensor(json_dict["token_with_gradients"]["token_ids"])
        token_with_gradients.gradients = Tensor(json_dict["token_with_gradients"]["gradients"])
        return InputImportanceCalculator(
            tokenizer,
            json_dict["prompt"],
            token_with_gradients
        )
