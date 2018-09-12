#!/usr/bin/env python

import sys
from setuptools import setup

if sys.version_info < (3, 5):
    raise NotImplementedError("Sorry, you need at least Python 3.5 to use tfserve.")

def readme():
    with open("README.md") as f:
        return f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(name="deepbrain",
      version="0.1",
      description="Deep Learning-based tools for processing brain images",
      long_description=readme(),
      long_description_content_type="text/markdown",
      author="Ivan Itzcovich",
      author_email="i.itzcovich@gmail.com",
      url="http://github.com/iitzco/deepbrain",
      keywords="deep-learning machine-learning tensorflow ai",
      scripts=["bin/deepbrain-extractor"],
      packages=["deepbrain"],
      license="MIT",
      platforms="any",
      install_requires=required, # Automatically download dependencies on requirements.txt
      python_requires=">3.5",
      classifiers=["Programming Language :: Python :: 3.5",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   ],
      include_package_data=True  # To include everything in MANIFEST.in
      )
