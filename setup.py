from setuptools import setup, find_packages
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
# Read the contents of your README file
with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neuroimager",
    version="0.0.8",
    description="A collection of utilities used for MRI data analysis",
    author="Wetiqe",
    author_email="jzni132134@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wetiqe/neuroimager",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas==1.5.3",
        "scipy==1.10.1",
        "pingouin",
        "seaborn",
        "sklearn",
        "nilearn",
        "networkx",
        "munkres",
        "scikit-network",
    ],
    extras_require={"torch": ["torch>=1.0.0"]},
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
