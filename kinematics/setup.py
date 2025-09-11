#!/usr/bin/env python3
"""
Setup script for Simplified Robot Planning Package
"""

from setuptools import setup, find_packages

setup(
    name="robot_planning_simple",
    version="1.0.0",
    description="Simplified robot motion planning with clean workflow",
    author="Thorn",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

