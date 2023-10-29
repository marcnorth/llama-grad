from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama_grad import LlamaGrad
from gradient_calculator.simple_gradient_calculator import SimpleGradientCalculator


def test_image():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    llama_grad = LlamaGrad(
        model,
        "Hello world",
        tokenizer=tokenizer,
        gradient_calculator=SimpleGradientCalculator()
    )
    llama_grad.generate(max_output_length=4)
    llama_grad.html_visualizer().all_outputs_to_image(
        output_dir="C:\\Users\\micro\\Documents\\phd\\Writing\\Requirement Engineering 2024\\test_dir"
    )

if __name__ == "__main__":
    test_image()