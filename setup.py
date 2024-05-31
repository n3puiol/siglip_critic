from setuptools import setup, find_packages

setup(
    name="video_language_critic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ftfy",
        "regex",
        "tqdm",
        "boto3",
        "requests",
        "pandas",
        "numpy",
        "gdown",
    ],
)
