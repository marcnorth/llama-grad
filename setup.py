from setuptools import setup, find_packages

setup(
    name="llama_grad_different_name",
    version="0.1",
    python_requires='>3.10',
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.34.0",
        "html2image>=2.0.4.3",
        "typing-extensions>=4.8.0"
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)
