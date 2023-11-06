import html
import os
from typing import List, Union, Optional
from html2image import Html2Image
from llama_grad.input_importance_calculator import GroupGradientPooling, MaxGradient, InputImportanceCalculator


class HtmlVisualizer:
    def __init__(self, importance_calculator: InputImportanceCalculator):
        self.importance_calculator = importance_calculator

    def nth_output_to_html(
            self,
            output_token_index: int,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None,
            prompt_only: bool = False,
            ignore: List[str] = []
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
        grouped_input_importance, grouped_input_token_ids = self.importance_calculator.calculate_importance_for_nth_output(output_token_index, groups=groups, group_gradient_pooling=group_gradient_pooling, max_gradient=max_gradient, prompt_only=prompt_only, ignore=ignore)
        grouped_input_tokens = [self.importance_calculator.tokenizer.decode(token) for token in grouped_input_token_ids]
        # Create html
        html_body = ""
        for input_token_ids, input_tokens, input_importance in zip(grouped_input_token_ids, grouped_input_tokens,
                                                                 grouped_input_importance):
            if not isinstance(input_token_ids, list):
                input_token_ids = [input_token_ids]
            html_body += f"<span data-token-ids=\"[{','.join([str(tid) for tid in input_token_ids])}]\" style=\"background-color:rgba(255,0,0,{input_importance:.3f})\">{html.escape(input_tokens)}</span>"
        css = """
                body {background-color:#fff}
                span[data-token-ids] {font-size: 20px;}
                """
        html_body = "<br/>".join(html_body.split("\n"))
        html_inc_head = f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
        return html_inc_head

    def nth_output_to_image(
            self,
            output_token_index: int,
            output_path: str,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None,
            prompt_only: bool = False,
            ignore: List[str] = []
    ):
        """
        Outputs the gradients of specified output token w.r.t. each input token
        By default 100% opacity is the max gradient
        :param output_token_index: The index of the output token
        :param output_path: Path to output image to
        :param max_gradient: The max value to use for calculating opacities. MaxGradient.SINGLE_OUTPUT (default) will use the maximum gradient for that output (i.e. that input will have opacity=1), MaxGradient.ALL_OUTPUTS will use the maximum across all output tokens
        :param groups: List of string to group gradients by. Gradients will be calculated according to group_gradient_pooling strategy
        :param group_gradient_pooling: How to pool the gradients of groups
        :param prompt_only: Only use the prompt input tokens and gradients, not the previous appended output tokens
        :param ignore: List of string to ignore in the input (e.g. special tokens)
        :return: html
        """
        html_str = self.nth_output_to_html(output_token_index, max_gradient, groups, group_gradient_pooling, prompt_only, ignore)
        dir_path, file_name = os.path.split(output_path)
        html2image = Html2Image(size=(800, 600), output_path=dir_path)
        html2image.screenshot(html_str=html_str, save_as=file_name)

    def all_outputs_to_image(
            self,
            output_dir: str,
            max_gradient: Union[MaxGradient, float] = MaxGradient.SINGLE_OUTPUT,
            groups: List[str] = [],
            group_gradient_pooling: Optional[GroupGradientPooling] = None,,
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
                output_path=os.path.join(output_dir, f"{i}.{file_extension}"),
                max_gradient=max_gradient,
                groups=groups,
                group_gradient_pooling=group_gradient_pooling,
                prompt_only=prompt_only,
                ignore=ignore
            )