import os
import sys
from setuptools import setup

setup(name ="kdtree",
    version = "1.0.0",
    description = "Python library for optimization descent algorithms",
    long_description = open("README.md").read(),
    url = "https://github.com/qzhao19/kdtree",
    author = 'Qi ZHAO',
    author_email = 'qi.zhao@outlook.fr',
    license = "MIT",
    packages = ['kdtree'],
    install_requires = ["numpy>=1.19.5"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics :: Machine Learning Algorithms",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        'Operating System :: Unix',
    ],
    keywords='kdtree',
)