from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read the contents of the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="neuroimager",
    version="0.0.5",
    description="A collection of utilities used for MRI data analysis",
    author="Wetiqe",
    author_email="jzni132134@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wetiqe/neuroimager",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
