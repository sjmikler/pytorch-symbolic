#  Copyright (c) 2022 Szymon Mikler

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pytorch-functional",
    version="0.4.10",
    url="https://github.com/gahaalt/pytorch-functional.git",
    project_urls={
        "Documentation": "https://pytorch-functional.readthedocs.io/",
    },
    author="Szymon Mikler",
    author_email="sjmikler@gmail.com",
    license="MIT",
    description="Provides functional API for model creation in PyTorch.",
    packages=["pytorch_functional"],
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=["torch>=1.9.0"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
