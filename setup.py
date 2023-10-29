from setuptools import setup

setup(
    name="llama_grad",
    version="0.1",
    python_requires='>3.11',
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.34.0",
        "html2image>=2.0.4.3"
    ]
)
