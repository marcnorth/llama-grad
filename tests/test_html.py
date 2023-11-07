from unittest import TestCase
import re
from torch import Tensor
from transformers import GPT2Tokenizer
from llama_grad import TokenWithGradients
from llama_grad.input_importance_calculator import MaxGradient, GroupGradientPooling, InputImportanceCalculator
from llama_grad.visualization import HtmlVisualizer

class TestHtml(TestCase):
    def test_create_html(self):
        """
        Creates html of input with visualisation of gradients
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one two"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 4., 0.],
            [3.3333333, 5., 10.]
        ])
        html_visualizer = HtmlVisualizer(InputImportanceCalculator(
            tokenizer,
            prompt="Hello world",
            token_with_gradients=token_with_gradients
        ))
        html = html_visualizer.nth_output_to_html(0)
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>([^<]*)</span>", html)
        self.assertEqual(2, len(matches))  # One for each input token
        self.assertEqual("Hello", matches[0][2])
        self.assertEqual("&nbsp;world", matches[1][2])
        self.assertEqual("0.250", matches[0][1])
        self.assertEqual("1.000", matches[1][1])
        html = html_visualizer.nth_output_to_html(1)
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>([^<]*)</span>", html)
        self.assertEqual("Hello", matches[0][2])
        self.assertEqual("&nbsp;world", matches[1][2])
        self.assertEqual("&nbsp;one", matches[2][2])
        self.assertEqual("0.333", matches[0][1])
        self.assertEqual("0.500", matches[1][1])
        self.assertEqual("1.000", matches[2][1])
        self.assertEqual(3, len(matches))  # One for each input token, plus one for the first output token

    def test_max_gradient(self):
        """
        Different max_gradient strategies give different opacities
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one two"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 4., 0.],
            [3.3333333, 5., 10.]
        ])
        html_visualizer = HtmlVisualizer(InputImportanceCalculator(
            tokenizer,
            prompt="Hello world",
            token_with_gradients=token_with_gradients
        ))
        # SINGLE_OUTPUT will use max for that output only (i.e. 4 = max([1., 4., 0.])
        html = html_visualizer.nth_output_to_html(
            0,
            max_gradient=MaxGradient.SINGLE_OUTPUT
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>([^<]*)</span>", html)
        self.assertEqual("0.250", matches[0][1])
        self.assertEqual("1.000", matches[1][1])
        # ALL_OUTPUTS will use max for all outputs (i.e. 10 = max([1., 4., 0., 3.333, 5., 10.])
        html = html_visualizer.nth_output_to_html(
            0,
            max_gradient=MaxGradient.ALL_OUTPUTS
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>([^<]*)</span>", html)
        self.assertEqual("0.100", matches[0][1])
        self.assertEqual("0.400", matches[1][1])
        # A float will just use that value as the max
        html = html_visualizer.nth_output_to_html(
            0,
            max_gradient=8.
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>([^<]*)</span>", html)
        self.assertEqual("0.125", matches[0][1])
        self.assertEqual("0.500", matches[1][1])

    def test_grouping(self):
        """
        Input tokens can be grouped when drawing gradients using different pooling strategies
        :return:
        """
        # Any input not in a group will be rendered individually
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus(["test"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
        html_visualizer = HtmlVisualizer(InputImportanceCalculator(
            tokenizer,
            prompt=" Group one Group two Not part Group one",
            token_with_gradients=token_with_gradients
        ))
        html = html_visualizer.nth_output_to_html(
            0,
            groups=[" Group one", " Group two", " Group one"]
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>(.*?)</span>", html)
        self.assertEqual(5, len(matches))
        self.assertEqual("&nbsp;Group&nbsp;one", matches[0][2])
        self.assertEqual("&nbsp;Group&nbsp;two", matches[1][2])
        self.assertEqual("&nbsp;Not", matches[2][2])
        self.assertEqual("&nbsp;part", matches[3][2])
        self.assertEqual("&nbsp;Group&nbsp;one", matches[4][2])
        # Group based on regex?

    def test_gradient_pooling(self):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus(["test"], return_tensors="pt", add_special_tokens=False)["input_ids"][
            0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
        html_visualizer = HtmlVisualizer(InputImportanceCalculator(
            tokenizer,
            prompt=" Group one Group two Not part Group one",
            token_with_gradients=token_with_gradients
        ))
        # Average pooling, max gradient is 7.5 (average of 7 and 8)
        html = html_visualizer.nth_output_to_html(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.AVERAGE
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>(.*?)</span>", html)
        self.assertEqual(5, len(matches))
        self.assertEqual("0.200", matches[0][1])  # = ((1 + 2) / 2) / 7.5 = 0.2
        self.assertEqual("0.467", matches[1][1])  # = ((3 + 4) / 2) / 7.5 = 0.467
        self.assertEqual("0.667", matches[2][1])  # = 5 / 7.5 = 0.667
        self.assertEqual("0.800", matches[3][1])  # = 6 / 7.5 = 0.8
        self.assertEqual("1.000", matches[4][1])  # = ((7 + 8) / 2) / 7.5 = 1
        # Max pooling, max gradient is 8
        html = html_visualizer.nth_output_to_html(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.MAX
        )
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>(.*?)</span>", html)
        self.assertEqual(5, len(matches))
        self.assertEqual("0.250", matches[0][1])  # = max(1,2) / 8 = 0.25
        self.assertEqual("0.500", matches[1][1])  # = max(3,4) / 8 = 0.5
        self.assertEqual("0.625", matches[2][1])  # = 5 / 8 = 0.625
        self.assertEqual("0.750", matches[3][1])  # = 6 / 8 = 0.75
        self.assertEqual("1.000", matches[4][1])  # = max(7,8) / 8 = 1

    def test_line_breaks(self):
        """
        Line breaks should be converted to <br/>
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 4.]])
        html_visualizer = HtmlVisualizer(InputImportanceCalculator(
            tokenizer,
            prompt="Hello>\nworld",
            token_with_gradients=token_with_gradients
        ))
        html = html_visualizer.nth_output_to_html(0)
        matches = re.findall("<span\s+data-token-ids=\"([\[\]0-9,]+)\"\s+style=\"background-color:rgba\([0-9\.]+,[0-9\.]+,[0-9\.]+,([0-9\.]+)\)\"[^>]*>(.*?)</span>", html)
        self.assertEqual(4, len(matches))
        self.assertEqual("Hello", matches[0][2])
        self.assertEqual("&gt;", matches[1][2])
        self.assertEqual("<br/>", matches[2][2])
        self.assertEqual("world", matches[3][2])
