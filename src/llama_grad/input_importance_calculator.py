from __future__ import annotations
import json
from enum import Enum
from statistics import mean
from typing import List, Tuple, Optional, Union, TextIO
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, AutoTokenizer, CodeLlamaTokenizerFast
import statistics

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
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False,
            exclude_z_scores_greater_than: float=None
    ) -> Tuple[List[float], List[int]]:
        """
        Calculates the importance of each input token for all outputs.
        Ignored tokens will have importance=0
        :return: Tuple[List of importance scores, List of token ids (or groups)]
        """
        input_ids = self._get_input_ids_for_calculation(prompt_only=True, ignore=ignore)
        input_gradients = self.token_with_gradients.gradients[:, :len(input_ids)].T.tolist()
        if exclude_z_scores_greater_than is not None:
            input_gradints_with_high_z_scores_removed = []
            for gradients_for_input in input_gradients:
                # Replace inf values
                max_val = max(val for val in gradients_for_input if val != float('Inf'))
                gradients_for_input = [val if val != float('Inf') else 2*max_val for val in gradients_for_input]
                average = sum(gradients_for_input) / len(gradients_for_input)
                standard_deviation = statistics.pstdev(gradients_for_input)
                filtered_on_z_score = [gradient for gradient in gradients_for_input if abs((gradient-average)/(standard_deviation if standard_deviation != 0 else 1)) <= exclude_z_scores_greater_than]
                input_gradints_with_high_z_scores_removed.append(filtered_on_z_score)
            input_gradients = input_gradints_with_high_z_scores_removed
        pooled_gradients = [mean(gradients)/len(gradients) for gradients in input_gradients] \
            if output_gradient_pooling == OutputGradientPooling.AVERAGE else \
            [max(gradients) for gradients in input_gradients]
        filtered_input_gradients = [gradient for gradient, input_id in zip(pooled_gradients, input_ids) if input_id is not None]
        return self._calculate_grouped_importance(input_ids, filtered_input_gradients, groups, group_gradient_pooling, max_gradient, ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens)

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
            found_index, group_token_ids = self._find_text_in_token_ids(group, input_ids, continue_searching_from)
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
            list(map(lambda g: g / max_input_gradient, grouped_input_gradients)) if max_input_gradient != 0 else [0] * len(grouped_input_gradients),
            grouped_input_ids
        )

    def _find_text_in_token_ids(self, text: str, token_ids: List[int], not_before: int = 0) -> Tuple[int, List[int]]:
        """
        Looks for the given token_ids in self.prompt, beginning at text_start_pos and returns the start index in prompt
        :param text: Text to look for
        :return: (Index in token_ids of the start of the text, The actual tokens that were found)
        """
        text_start_poses = self._find_all_substrings(self.prompt, text)
        if len(text_start_poses) == 0:
            raise ValueError(f"Text '{text}' not found in text '{self.prompt}'")
        for text_start_pos in text_start_poses:
            looking_from = text_start_pos
            looking_to = looking_from + len(text)
            len_of_token_sequence_to_look_for = len(self.tokenizer.encode(self.prompt[looking_from:looking_to]))
            while True:
                prompt_segment_to_look_for = self.prompt[looking_from:looking_to]
                segment_token_ids = self.tokenizer.encode(prompt_segment_to_look_for, add_special_tokens=False)
                if segment_token_ids[0] == 29871 and type(self.tokenizer) == CodeLlamaTokenizerFast: # See https://github.com/huggingface/transformers/issues/26273
                    segment_token_ids = segment_token_ids[1:]
                if (len(segment_token_ids) > len_of_token_sequence_to_look_for):
                    if looking_to == looking_from + len(text) or looking_from == 0:
                        # We reach the end when segment is too long on the first shift back
                        break
                    looking_from -= 1
                    looking_to = looking_from + len(text) + 1
                    continue
                token_start_indexes = self._find_sublist(segment_token_ids, token_ids)
                if len(token_start_indexes) == 0:
                    if looking_to > len(token_ids):
                        looking_from -= 1
                        looking_to = looking_from + len(text) + 1
                    else:
                        looking_to += 1
                    continue
                # Return first token_start_index that isn't less than not_before
                for token_start_index in token_start_indexes:
                    if token_start_index >= not_before:
                        return (token_start_index, segment_token_ids)
                break
        raise Exception(f"Text '{text}' not found in tokens for '{self.prompt}'")

    def _find_all_substrings(self, haystack: str, needle: str) -> list[int]:
        start = 0
        found = []
        while True:
            start = haystack.find(needle, start)
            if start == -1:
                return found
            found.append(start)
            start += len(needle)

    def _find_sublist(self, sublist: List[int], list: List[int]) -> List[int]:
        """
        Looks for a sublist in a list
        :param sublist:
        :param list:
        :return List of the indexes of the start sublist in the list, [] if no sublist is found
        """
        sublist_length = len(sublist)
        found = []
        if sublist_length == 0:
            return []
        checking_index = 0
        while checking_index <= len(list) - sublist_length:
            if list[checking_index:checking_index+sublist_length] == sublist:
                found.append(checking_index)
                checking_index += sublist_length
            else:
                checking_index += 1
        return found

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
        token_with_gradients.token_ids = torch.IntTensor(json_dict["token_with_gradients"]["token_ids"])
        token_with_gradients.gradients = torch.FloatTensor(json_dict["token_with_gradients"]["gradients"])
        return InputImportanceCalculator(
            tokenizer,
            json_dict["prompt"],
            token_with_gradients
        )
