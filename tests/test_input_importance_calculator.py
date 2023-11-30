import tempfile
from unittest import TestCase
from torch import Tensor
from transformers import GPT2Tokenizer
from llama_grad import TokenWithGradients
from llama_grad.input_importance_calculator import InputImportanceCalculator, MaxGradient, GroupGradientPooling, \
    OutputGradientPooling


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
        self.assertAlmostEqual(0.25, importance_scores[0]) # = 1. / 4
        self.assertAlmostEqual(1., importance_scores[1]) # = 4. / 4
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(1)
        self.assertAlmostEqual(3, len(importance_scores))  # One for each input token
        self.assertAlmostEqual(1./3., importance_scores[0]) # = 3.333 / 10
        self.assertAlmostEqual(0.5, importance_scores[1]) # = 5. / 10
        self.assertAlmostEqual(1., importance_scores[2]) # = 10. / 10
        # All outputs (Average pooling)
        importance_scores, _ = importance_calculator.calculate_importance_for_all_outputs()
        self.assertAlmostEqual(2, len(importance_scores))  # One for each input token (all_outputs only makes sense with prompt_only=True)
        self.assertAlmostEqual(0.481481446, importance_scores[0]) # Each gradient is averaged across all outputs = AVERAGE(1,3.33333) / AVERAGE(4,5)
        self.assertAlmostEqual(1., importance_scores[1]) # Each gradient is averaged across all outputs = AVG(4,5) / AVG(4,5)
        # All outputs (Max pooling)
        importance_scores, _ = importance_calculator.calculate_importance_for_all_outputs(output_gradient_pooling=OutputGradientPooling.MAX)
        self.assertAlmostEqual(2, len(importance_scores))
        self.assertAlmostEqual(0.66666666, importance_scores[0]) # Each gradient is maxed across all outputs = MAX(1,3.33333) / MAX(4,5)
        self.assertAlmostEqual(1., importance_scores[1]) # Each gradient is maxed across all outputs = MAX(4,5) / MAX(4,5)


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

    def test_ignore_non_grouped(self):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus(["test"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 4., 5., 8., 7., 6.]])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt=" Group one Group two Not part Group one",
            token_with_gradients=token_with_gradients
        )
        # Don't ignore non-grouped tokens (Max gradient is 8)
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.AVERAGE,
            ignore_non_grouped_input_tokens=False
        )
        self.assertAlmostEqual(5, len(importance_scores))
        self.assertAlmostEqual(0.1875, importance_scores[0])  # = ((1 + 2) / 2) / 8 = 0.
        self.assertAlmostEqual(7. / 16., importance_scores[1])  # = ((3 + 4) / 2) / 8 = 0.467
        self.assertAlmostEqual(0.625, importance_scores[2])  # = 5 / 8 = 0.625
        self.assertAlmostEqual(1., importance_scores[3])  # = 8 / 8 = 1
        self.assertAlmostEqual(0.8125, importance_scores[4])  # = 6.5 / 8 = 1
        # Ignore non-grouped tokens (Max gradient is 6.5)
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            groups=[" Group one", " Group two", " Group one"],
            group_gradient_pooling=GroupGradientPooling.AVERAGE,
            ignore_non_grouped_input_tokens=True
        )
        print(importance_scores)
        self.assertAlmostEqual(5, len(importance_scores))
        self.assertAlmostEqual(0.23076923, importance_scores[0])  # = ((1 + 2) / 2) / 6.5 = 0.2
        self.assertAlmostEqual(0.53846154, importance_scores[1])  # = ((3 + 4) / 2) / 6.5 = 0.467
        self.assertAlmostEqual(0., importance_scores[2])  # Ignored
        self.assertAlmostEqual(0., importance_scores[3])  # Ignored
        self.assertAlmostEqual(1., importance_scores[4])  # = 6.5 / 6.5 = 1

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

    def test_ignore_values(self):
        """
        Can ignore groups in the input
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = \
        tokenizer.batch_encode_plus(["test"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([[1., 2., 3., 8., 5., 6., 7., 4.]])
        importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt="code will ignore this but not this ignore",
            token_with_gradients=token_with_gradients
        )
        # Default will use all inputs
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(0)
        self.assertEqual(8, len(importance_scores))
        self.assertAlmostEqual(1./8., importance_scores[0])
        self.assertAlmostEqual(3./8., importance_scores[2])
        # Ignoring 'ignore this' and 'not' will remove those tokens from the calculated importance scores
        importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(
            0,
            ignore=[" ignore this", " not"]
        )
        self.assertEqual(5, len(importance_scores))
        self.assertAlmostEqual(1./7., importance_scores[0])  # 1. is first gradient and 7. is max gradient once 'this' is ignored
        self.assertAlmostEqual(5./7., importance_scores[2])  # 5. is third gradient and 7. is max gradient once 'this' is ignored

    def test_save_and_load(self):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        output_ids = tokenizer.batch_encode_plus([" one two"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        token_with_gradients = TokenWithGradients()
        token_with_gradients.token_ids = output_ids
        token_with_gradients.gradients = Tensor([
            [1., 4., 0.],
            [3.3333333, 5., 10.]
        ])
        orig_importance_calculator = InputImportanceCalculator(
            tokenizer,
            prompt="Hello world",
            token_with_gradients=token_with_gradients
        )
        # Save then load
        with tempfile.TemporaryFile("w+") as file:
            orig_importance_calculator.save(file)
            file.seek(0)
            importance_calculator = InputImportanceCalculator.load(file)
            importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(0)
            self.assertAlmostEqual(2, len(importance_scores))  # One for each input token
            self.assertAlmostEqual(0.25, importance_scores[0])
            self.assertAlmostEqual(1., importance_scores[1])
            importance_scores, _ = importance_calculator.calculate_importance_for_nth_output(1)
            self.assertAlmostEqual(3, len(importance_scores))  # One for each input token
            self.assertAlmostEqual(1. / 3., importance_scores[0])
            self.assertAlmostEqual(0.5, importance_scores[1])
            self.assertAlmostEqual(1., importance_scores[2])
