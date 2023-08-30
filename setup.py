from setuptools import setup, find_packages

setup(
    name="my-ai-package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "joblib~=1.3.2",
        "numpy~=1.24.2",
        "setuptools~=65.5.0",
        "scikit-learn~=1.3.0",
        "pandas~=2.0.3",
        "tensorflow~=2.13.0"
    ],
)
