from unittest import TestCase
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from core import LlamaGrad
from token_with_gradients import TokenWithGradients
from gradient_calculator.simple_gradient_calculator import SimpleGradientCalculator


class TestLlamaGrad(TestCase):
    def test_next_token(self):
        """
        Inputting size n Tensor (n input_token_ids) returns next token ID for each batch and Tensor of size (1, n) for gradients. 1, since output_sequence_length is 1
        :return:
        """
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        llama_grad = LlamaGrad(
            model,
            "Hello world",
            tokenizer=tokenizer,
            gradient_calculator=SimpleGradientCalculator()
        )
        result = llama_grad.next_token()
        self.assertIsInstance(result, TokenWithGradients)
        self.assertEqual((1,), result.token_ids.shape)  # one output token
        self.assertEqual((1, 2), result.gradients.shape)  # one output token, two gradient magnitudes for each of the two input token

    def test_next_token_twice(self):
        """
        Each time we call next_token it should have an extra gradient, since the input grows by 1 each time it's called (as previous output is appended to input)
        :return:
        """
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        llama_grad = LlamaGrad(
            model,
            "Hello world",
            tokenizer=tokenizer,
            gradient_calculator=SimpleGradientCalculator()
        )
        result = llama_grad.next_token()
        self.assertIsInstance(result, TokenWithGradients)
        self.assertEqual((1,), result.token_ids.shape)
        self.assertEqual((1, 2), result.gradients.shape)
        self.assertEqual((1,), llama_grad.output_tokens_with_gradients.token_ids.shape)
        self.assertEqual((1, 2), llama_grad.output_tokens_with_gradients.gradients.shape)
        result2 = llama_grad.next_token()
        self.assertIsInstance(result2, TokenWithGradients)
        self.assertEqual((1,), result2.token_ids.shape)
        self.assertEqual((1, 3), result2.gradients.shape)
        self.assertEqual((2,), llama_grad.output_tokens_with_gradients.token_ids.shape)
        self.assertEqual((2, 3), llama_grad.output_tokens_with_gradients.gradients.shape)
        result3 = llama_grad.next_token()
        self.assertIsInstance(result3, TokenWithGradients)
        self.assertEqual((1,), result3.token_ids.shape)
        self.assertEqual((1, 4), result3.gradients.shape)
        self.assertEqual((3,), llama_grad.output_tokens_with_gradients.token_ids.shape)
        self.assertEqual((3, 4), llama_grad.output_tokens_with_gradients.gradients.shape)

    def test_generate(self):
        """
        Inputting size n Tensor (n input_token_ids) returns all token IDs and Tensor of size (m, n) for gradients. m is the number of output tokens
        :return:
        """
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        llama_grad = LlamaGrad(
            model,
            "Hello world",
            tokenizer=tokenizer,
            gradient_calculator=SimpleGradientCalculator()
        )
        result = llama_grad.generate(max_output_length=3)
        self.assertIsInstance(result, TokenWithGradients)
        self.assertEqual((3,), result.token_ids.shape)  # three output tokens
        self.assertEqual((3, 4), result.gradients.shape)  # three output tokens, two gradient magnitudes for each of the two input tokens and two for the first two outputs
        self.assertEqual((3,), llama_grad.output_tokens_with_gradients.token_ids.shape)
        self.assertEqual((3, 4), llama_grad.output_tokens_with_gradients.gradients.shape)
        result2 = llama_grad.generate(max_output_length=4)
        self.assertIsInstance(result2, TokenWithGradients)
        self.assertEqual((4,), result2.token_ids.shape)  # four output tokens
        self.assertEqual((4, 8), result2.gradients.shape)  # five output tokens, 4 gradient magnitudes for each of the accumulated input tokens and three for the new outputs
        self.assertEqual((7,), llama_grad.output_tokens_with_gradients.token_ids.shape)
        self.assertEqual((7, 8), llama_grad.output_tokens_with_gradients.gradients.shape)
