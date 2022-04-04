#!/usr/bin/env python
import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sgd_env",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["sgd_env",
              "sgd_env.envs"],
    install_requires=["gym",
                      "ConfigSpace",
                      "numpy",
                      "torch",
                      "torchvision"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
