import html
import os
from enum import Enum
from statistics import mean
from typing import List, Union, Tuple, Optional
import torch
from html2image import Html2Image
from transformers import PreTrainedTokenizerBase
from token_with_gradients import TokenWithGradients


class MaxGradient(Enum):
    ALL_OUTPUTS = 1  # Uses max gradient across all outputs
    SINGLE_OUTPUT = 2  # Uses max gradient of only the output html is being generated for


class GroupGradientPooling(Enum):
    AVERAGE = 1
    MAX = 2


class HtmlVisualizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt: str, token_with_gradients: TokenWithGradients):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.input_token_ids: List[int] = tokenizer.batch_encode_plus([prompt], add_special_tokens=False)["input_ids"][
            0]
        self.input_token_count = len(self.input_token_ids)
        self.token_with_gradients = token_with_gradients

    def nth_output_to_html(
            self,
            output_token_index: int,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None
    ) -> str:
        """
        Outputs the gradients of specified output token w.r.t. each input token
        By default 100% opacity is the max gradient
        :param output_token_index: The index of the output token
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :return: html
        """
        # We only want gradient values up to this output token's position
        grouped_input_token_ids, grouped_input_gradients = self._group_input_token_ids(output_token_index, groups=groups, group_gradient_pooling=group_gradient_pooling)
        grouped_input_tokens = [self.tokenizer.decode(token) for token in grouped_input_token_ids]
        output_token_id = self.token_with_gradients.token_ids[output_token_index].item()
        output_token = self.tokenizer.decode(output_token_id)
        # remaining_input_text = input_text
        max_input_gradient = (
            max(grouped_input_gradients) if max_gradient == MaxGradient.SINGLE_OUTPUT else
            torch.max(self.token_with_gradients.gradients) if max_gradient == MaxGradient.ALL_OUTPUTS else
            max_gradient)
        # Create html
        html_body = ""
        for input_token_ids, input_tokens, input_gradient in zip(grouped_input_token_ids, grouped_input_tokens,
                                                                 grouped_input_gradients):
            opacity = input_gradient / max_input_gradient
            if not isinstance(input_token_ids, list):
                input_token_ids = [input_token_ids]
            html_body += f"<span data-token-ids=\"[{','.join([str(tid) for tid in input_token_ids])}]\" style=\"background-color:rgba(255,0,0,{opacity:.3f})\">{html.escape(input_tokens)}</span>"
        css = """
                body {background-color:#fff}
                span[data-token-ids] {font-size: 20px;}
                """
        html_body = "<br/>".join(html_body.split("\n"))
        html_inc_head = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        return html_inc_head

    def _group_input_token_ids(self, output_token_index: int, groups: List[str] = None,
                               group_gradient_pooling=Optional[GroupGradientPooling]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Groups the input token ids based on the given groups. Any tokens not in a group are returned individually
        :param output_token_index:
        :param groups:
        :return:
        """
        input_ids = self.input_token_ids + self.token_with_gradients.token_ids[:output_token_index].tolist()
        input_gradients = self.token_with_gradients.gradients[output_token_index][
                          :self.input_token_count + output_token_index]
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
            group_gradient = mean(input_gradients[found_index:found_index + len(group_token_ids)].tolist()) if group_gradient_pooling == GroupGradientPooling.AVERAGE else max(input_gradients[found_index:found_index + len(group_token_ids)].tolist())
            grouped_input_gradients.append(group_gradient)
            continue_searching_from = found_index + len(group_token_ids)
        # Add any tokens not already added
        grouped_input_ids.extend(input_ids[continue_searching_from:])
        grouped_input_gradients.extend(input_gradients[continue_searching_from:].tolist())
        return grouped_input_ids, grouped_input_gradients

    def nth_output_to_image(
            self,
            output_token_index: int,
            output_path: str,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None
    ):
        """
        Outputs the gradients of specified output token w.r.t. each input token
        By default 100% opacity is the max gradient
        :param output_token_index: The index of the output token
        :param output_path: Path to output image to
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :return: html
        """
        html_str = self.nth_output_to_html(output_token_index, max_gradient, groups, group_gradient_pooling)
        dir_path, file_name = os.path.split(output_path)
        html2image = Html2Image(size=(800, 600), output_path=dir_path)
        html2image.screenshot(html_str=html_str, save_as=file_name)

    def all_outputs_to_image(
            self,
            output_dir: str,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None,
            file_extension: str = 'png'
    ):
        """
        Calls self.nth_output_to_image for all outputs, saving one image per output in the given output_dir
        :param output_dir: Directory will be created if it doesn't exist
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :return: html
        :return:
        """
        for i in range(self.token_with_gradients.token_ids.shape[0]):
            self.nth_output_to_image(
                i,
                output_path=os.path.join(output_dir, f"{i}.{file_extension}"),
                max_gradient=max_gradient,
                groups=groups,
                group_gradient_pooling=group_gradient_pooling
            )