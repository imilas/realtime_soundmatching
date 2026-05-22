"""Multi-dimensional optimization agents for the sound-matching paper."""

from .base import MultiDimAgentBase, Bounds
from .random_search import MultiDimRandomSearch
from .hill_climber import MultiDimHillClimber
from .cma_es import CMAESAgent
from .bayesian_opt import BayesianOptAgent

__all__ = [
    "MultiDimAgentBase",
    "Bounds",
    "MultiDimRandomSearch",
    "MultiDimHillClimber",
    "CMAESAgent",
    "BayesianOptAgent",
]
