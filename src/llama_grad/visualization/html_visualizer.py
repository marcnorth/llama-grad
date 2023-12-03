import html
import re
from typing import List, Union, Optional
from html2image import Html2Image
from llama_grad.input_importance_calculator import GroupGradientPooling, MaxGradient, InputImportanceCalculator, \
    OutputGradientPooling


class InputNotFoundException(Exception):
    pass


class HtmlVisualizer:
    def __init__(self, importance_calculator: InputImportanceCalculator):
        self.importance_calculator = importance_calculator

    def nth_output_to_html(
            self,
            output_token_index: int,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: GroupGradientPooling = GroupGradientPooling.AVERAGE,
            prompt_only: bool = False,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False
    ) -> str:
        """
        Outputs the gradients of specified output token w.r.t. each input token
        By default 100% opacity is the max gradient
        :param output_token_index: The index of the output token
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: html
        """
        grouped_input_importance, grouped_input_token_ids = self.importance_calculator.calculate_importance_for_nth_output(output_token_index, groups=groups, group_gradient_pooling=group_gradient_pooling, max_gradient=max_gradient, prompt_only=prompt_only, ignore=ignore, ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens)
        return self._grouped_input_importance_to_html(grouped_input_importance, grouped_input_token_ids)

    def all_outputs_to_html(
            self,
            output_gradient_pooling: OutputGradientPooling = OutputGradientPooling.AVERAGE,
            groups: List[str] = [],
            group_gradient_pooling: GroupGradientPooling = GroupGradientPooling.AVERAGE,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False
    ) -> str:
        grouped_input_importance, grouped_input_token_ids = self.importance_calculator.calculate_importance_for_all_outputs(output_gradient_pooling=output_gradient_pooling, groups=groups, group_gradient_pooling=group_gradient_pooling, ignore=ignore, ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens)
        return self._grouped_input_importance_to_html(grouped_input_importance, grouped_input_token_ids)

    def _grouped_input_importance_to_html(self, grouped_input_importance: List[float], grouped_input_token_ids: List[int]) -> str:
        grouped_input_tokens = [self.importance_calculator.tokenizer.decode(token) for token in grouped_input_token_ids]
        flattened_input_token_ids = self._flattern_input_token_ids(grouped_input_token_ids)
        whole_input = self.importance_calculator.tokenizer.decode(flattened_input_token_ids)
        remaining_input = whole_input
        # Create html
        html_body = ""
        for input_token_ids, input_tokens, input_importance in zip(grouped_input_token_ids, grouped_input_tokens, grouped_input_importance):
            if not isinstance(input_token_ids, list):
                input_token_ids = [input_token_ids]
            found = False
            for n_pad in range(4):
                search_string = f"{' ' * n_pad}{input_tokens}"
                if remaining_input.startswith(search_string):
                    remaining_input = remaining_input[len(search_string):]
                    found = True
                    html_body += f"<span data-token-ids=\"[{','.join([str(tid) for tid in input_token_ids])}]\" style=\"background-color:rgba(255,0,0,{input_importance:.3f})\">{html.escape(search_string).replace(' ', '&nbsp;')}</span>"
            if not found:
                raise InputNotFoundException(f"Input '{input_tokens}' not found. remaining input:'{remaining_input}'")
        css = """
                body {background-color:#fff}
                span[data-token-ids] {font-size: 20px;}
                """
        html_body = "&#8629;<br/>".join(html_body.split("\n"))
        html_inc_head = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        return html_inc_head

    def _flattern_input_token_ids(self, grouped_input_token_ids) -> List[int]:
        flattened_input_token_ids = []
        for input_token_ids in grouped_input_token_ids:
            if isinstance(input_token_ids, list):
                flattened_input_token_ids.extend(input_token_ids)
            else:
                flattened_input_token_ids.append(input_token_ids)
        return flattened_input_token_ids

    def all_outputs_to_image(
            self,
            output_dir: str,
            output_file_name: Optional[str] = None,
            output_gradient_pooling: OutputGradientPooling = OutputGradientPooling.AVERAGE,
            groups: List[str] = [],
            group_gradient_pooling: GroupGradientPooling = GroupGradientPooling.AVERAGE,
            ignore: List[str] = [],
            ignore_non_grouped_input_tokens: bool = False
    ) -> None:
        if output_file_name is None:
            output_file_name = "all_outputs.png"
        html_str = self.all_outputs_to_html(
            output_gradient_pooling=output_gradient_pooling,
            groups=groups,
            group_gradient_pooling=group_gradient_pooling,
            ignore=ignore,
            ignore_non_grouped_input_tokens=ignore_non_grouped_input_tokens
        )
        html2image = Html2Image(size=(800, 600), output_path=output_dir)
        html2image.screenshot(html_str=html_str, save_as=output_file_name)

    def nth_output_to_image(
            self,
            output_token_index: int,
            output_dir: str,
            output_file_name: Optional[str] = None,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: GroupGradientPooling = GroupGradientPooling.AVERAGE,
            prompt_only: bool = False,
            ignore: List[str] = []
    ):
        """
        Outputs the gradients of specified output token w.r.t. each input token
        By default 100% opacity is the max gradient
        :param output_token_index: The index of the output token
        :param output_dir:
        :param output_file_name:
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: html
        """
        if output_file_name is None:
            nth_token_id = self.importance_calculator.token_with_gradients.token_ids[output_token_index].item()
            nth_token = self.importance_calculator.tokenizer.decode(nth_token_id)
            if re.search(r"[^A-Za-z0-9 ()_\-,.]$", nth_token):
                nth_token = f"token_{output_token_index}"
            output_file_name = f"{output_token_index}_{nth_token.strip()}.png"
        html_str = self.nth_output_to_html(
            output_token_index,
            max_gradient=max_gradient,
            groups=groups,
            group_gradient_pooling=group_gradient_pooling,
            prompt_only=prompt_only,
            ignore=ignore
        )
        html2image = Html2Image(size=(800, 600), output_path=output_dir)
        html2image.screenshot(html_str=html_str, save_as=output_file_name)

    def every_outputs_to_image(
            self,
            output_dir: str,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: GroupGradientPooling = GroupGradientPooling.AVERAGE,
            prompt_only: bool = False,
            ignore: List[str] = [],
            file_extension: str = 'png'
    ):
        """
        Calls self.nth_output_to_image for all outputs, saving one image per output in the given output_dir
        :param output_dir: Directory will be created if it doesn't exist
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: html
        :return:
        """
        for i in range(self.importance_calculator.token_with_gradients.token_ids.shape[0]):
            self.nth_output_to_image(
                i,
                output_dir=output_dir,
                max_gradient=max_gradient,
                groups=groups,
                group_gradient_pooling=group_gradient_pooling,
                prompt_only=prompt_only,
                ignore=ignore
            )