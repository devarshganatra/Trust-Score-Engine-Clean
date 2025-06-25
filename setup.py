#!/usr/bin/env python3
"""
Setup script for Trust Score Engine
"""

import os
import sys
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="trust-score-engine",
    version="1.0.0",
    description="A comprehensive review credibility assessment system using AI/ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Trust Score Engine Team",
    author_email="team@trustscoreengine.com",
    url="https://github.com/trustscoreengine/trust-score-engine",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "trust-engine=main:main",
        ],
    },
    keywords="trust score, review analysis, machine learning, nlp, sentiment analysis",
    project_urls={
        "Bug Reports": "https://github.com/trustscoreengine/trust-score-engine/issues",
        "Source": "https://github.com/trustscoreengine/trust-score-engine",
        "Documentation": "https://trustscoreengine.readthedocs.io/",
    },
) 