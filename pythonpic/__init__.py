# coding=utf-8
"""A work in progress particle-in-cell code written in Python, optimized for speed as well as readability."""

from .helper_functions.helpers import plotting_parser
from .algorithms import BoundaryCondition
import sys
import warnings

if sys.version_info[:2] < (3, 6):
    warnings.warn("PythonPIC does not support Python 3.5 and below")