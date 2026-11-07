"""Exact small-data RBF Gaussian-process regression."""
from .core import fit,predict
from .acquisition import expected_improvement,suggest
__all__=['fit','predict','expected_improvement','suggest']
