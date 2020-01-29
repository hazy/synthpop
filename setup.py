from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("VERSION", "r") as f:
    version = f.read().strip()

setup(
    name="py-synthpop",
    version=version,
    author="Georgi Ganev, Sofiane Mahiou",
    author_email="info@hazy.com",
    description="Python implementation of the R package synthpop",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hazy/synthpop",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
