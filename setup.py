#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="pelphix",
    version="0.0.0",
    description="Pelphix: Surgical Phase Recognition from X-ray Images in Percutaneous Pelvic Fixation",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/pelphix",
    install_requires=[
        "torch",
        "torchvision",
        "lightning",
        "hydra-core",
        "omegaconf",
        "rich",
        "numpy",
        "deepdrr",
        "trimesh",
        "gpustat",
        "strenum",
        "shapely",
        "stringcase",
        "perphix",
    ],
    packages=find_packages(),
    package_dir={"": "src"},
)
