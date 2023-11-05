from enum import Enum
from statistics import mean
from typing import List, Tuple, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from llama_grad import TokenWithGradients


class GroupGradientPooling(Enum):
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

    def calculate_importance_for_nth_output(
            self,
            output_token_index: int,
            groups: List[str] = [],
            group_gradient_pooling=Optional[GroupGradientPooling],
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT
    ) -> Tuple[List[float], List[int]]:
        """
        Calculates the importance of each input token for the nth output token
        :return: Tuple[List of importance scores, List of token ids (or groups)]
        """
        input_ids = self.input_token_ids + self.token_with_gradients.token_ids[:output_token_index].tolist()
        input_gradients = self.token_with_gradients.gradients[output_token_index][:self.input_token_count + output_token_index]
        grouped_input_ids = []
        grouped_input_gradients = []
        continue_searching_from = 0  # Index to continue searching for group from
        for group in groups:
            # Look for group in input_ids
            group_token_ids = self.tokenizer.encode_plus(group)["input_ids"]
            found_index = None
            for i in range(continue_searching_from, len(input_ids) - len(group_token_ids) + 1):
                if input_ids[i:i + len(group_token_ids)] == group_token_ids:
                    found_index = i
                    break
            if found_index is None:
                raise ValueError(f"Group '{group}' not found")
            # Add all individual tokens before group, then add group
            if found_index > continue_searching_from:
                grouped_input_ids.extend(input_ids[continue_searching_from:found_index])
                grouped_input_gradients.extend(input_gradients[continue_searching_from:found_index].tolist())
            grouped_input_ids.append(group_token_ids)
            # Calculate group gradient using pooling strategy
            group_gradient = mean(input_gradients[found_index:found_index + len(
                group_token_ids)].tolist()) if group_gradient_pooling == GroupGradientPooling.AVERAGE else max(
                input_gradients[found_index:found_index + len(group_token_ids)].tolist())
            grouped_input_gradients.append(group_gradient)
            continue_searching_from = found_index + len(group_token_ids)
        # Add any tokens not already added
        grouped_input_ids.extend(input_ids[continue_searching_from:])
        grouped_input_gradients.extend(input_gradients[continue_searching_from:].tolist())
        max_input_gradient = (
            max(grouped_input_gradients) if max_gradient == MaxGradient.SINGLE_OUTPUT else
            torch.max(self.token_with_gradients.gradients) if max_gradient == MaxGradient.ALL_OUTPUTS else
            max_gradient)
        return (
            list(map(lambda g: g / max_input_gradient, grouped_input_gradients)),
            grouped_input_ids
        )
