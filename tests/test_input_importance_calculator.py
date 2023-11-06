from unittest import TestCase
from torch import Tensor
from transformers import GPT2Tokenizer
from llama_grad import TokenWithGradients
from llama_grad.input_importance_calculator import InputImportanceCalculator, MaxGradient, GroupGradientPooling


class TestInputImportanceCalculator(TestCase):
    def test_basic_calculation(self):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one two"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 4., 0.],
            [3.3333333, 5., 10.]
        ])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt="Hello world",
            token_with_gradients=token_with_gradients
        )
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(0)
        self.assertAlmostEqual(2, len(importance_scores))  # One for each input token
        self.assertAlmostEqual(0.25, importance_scores[0])
        self.assertAlmostEqual(1., importance_scores[1])
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(1)
        self.assertAlmostEqual(3, len(importance_scores))  # One for each input token
        self.assertAlmostEqual(1./3., importance_scores[0])
        self.assertAlmostEqual(0.5, importance_scores[1])
        self.assertAlmostEqual(1., importance_scores[2])

    def test_max_gradient(self):
        """
        Different max_gradient strategies give different importance
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one two"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 4., 0.],
            [3.3333333, 5., 10.]
        ])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt="Hello world",
            token_with_gradients=token_with_gradients
        )
        # SINGLE_OUTPUT will use max for that output only (i.e. 4 = max([1., 4., 0.])
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            max_gradient=MaxGradient.SINGLE_OUTPUT
        )
        self.assertAlmostEqual(0.25, importance_scores[0])
        self.assertAlmostEqual(1., importance_scores[1])
        # ALL_OUTPUTS will use max for all outputs (i.e. 10 = max([1., 4., 0., 3.333, 5., 10.])
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            max_gradient=MaxGradient.ALL_OUTPUTS
        )
        self.assertAlmostEqual(0.1, importance_scores[0])
        self.assertAlmostEqual(0.4, importance_scores[1])
        # A float will just use that value as the max
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            max_gradient=8.
        )
        self.assertAlmostEqual(0.125, importance_scores[0])
        self.assertAlmostEqual(0.5, importance_scores[1])
        
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
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt=" Group one Group two Not part Group one",
            token_with_gradients=token_with_gradients
        )
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            groups=[" Group one", " Group two", " Group one"]
        )
        self.assertAlmostEqual(5, len(importance_scores))

    def test_gradient_pooling(self):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus(["test"], return_tensors="pt", add_special_tokens=False)["input_ids"][
            0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt=" Group one Group two Not part Group one",
            token_with_gradients=token_with_gradients
        )
        # Average pooling, max gradient is 7.5 (average of 7 and 8)
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.AVERAGE
        )
        self.assertAlmostEqual(5, len(importance_scores))
        self.assertAlmostEqual(0.2, importance_scores[0])  # = ((1 + 2) / 2) / 7.5 = 0.2
        self.assertAlmostEqual(7./15., importance_scores[1])  # = ((3 + 4) / 2) / 7.5 = 0.467
        self.assertAlmostEqual(2./3., importance_scores[2])  # = 5 / 7.5 = 0.667
        self.assertAlmostEqual(0.8, importance_scores[3])  # = 6 / 7.5 = 0.8
        self.assertAlmostEqual(1., importance_scores[4])  # = ((7 + 8) / 2) / 7.5 = 1
        # Max pooling, max gradient is 8
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.MAX
        )
        self.assertAlmostEqual(5, len(importance_scores))
        self.assertAlmostEqual(0.25, importance_scores[0])  # = max(1,2) / 8 = 0.25
        self.assertAlmostEqual(0.5, importance_scores[1])  # = max(3,4) / 8 = 0.5
        self.assertAlmostEqual(0.625, importance_scores[2])  # = 5 / 8 = 0.625
        self.assertAlmostEqual(0.75, importance_scores[3])  # = 6 / 8 = 0.75
        self.assertAlmostEqual(1., importance_scores[4])  # = max(7,8) / 8 = 1

    def test_prompt_only(self):
        """
        Calculates importance and max using ONLY the original prompt
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus(["test prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 2., 3., 4., 5., 6., 0.],
            [1., 2., 3., 4., 5., 6., 7.],
        ])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt="one two three four five six",
            token_with_gradients=token_with_gradients
        )
        # Default of 2nd token includes the appended 1st output token
        default_importance_scores_0, _ = importance_calculator.calculate_importance_for_nth_output(0)
        default_importance_scores_1, _ = importance_calculator.calculate_importance_for_nth_output(1)
        self.assertEqual(6, len(default_importance_scores_0))
        self.assertEqual(7, len(default_importance_scores_1))
        self.assertAlmostEqual(1./7., default_importance_scores_1[0])
        # prompt_only of 2nd token does not include the appended 1st output token
        prompt_only_importance_scores_0, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            prompt_only=True
        )
        prompt_only_importance_scores_1, _ = importance_calculator.calculate_importance_for_nth_output(
            1,
            prompt_only=True
        )
        self.assertEqual(6, len(prompt_only_importance_scores_0))
        self.assertEqual(6, len(prompt_only_importance_scores_1))
        self.assertAlmostEqual(1./6., prompt_only_importance_scores_1[0])
