from setuptools import setup, find_packages

setup(
    name='neuroimager',
    version='0.0.1',
    description='A collection of utilities used for MRI data analysis',
    author='Wetiqe@GitHub',
    author_email='jzni132134@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'sklearn',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

