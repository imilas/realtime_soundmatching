"""Multi-dimensional optimization agents for the sound-matching paper."""

from .base import MultiDimAgentBase, Bounds
from .random_search import MultiDimRandomSearch
from .hill_climber import MultiDimHillClimber
from .cma_es import CMAESAgent
from .bayesian_opt import BayesianOptAgent
from .q_learning import MultiDimQLearning

__all__ = [
    "MultiDimAgentBase",
    "Bounds",
    "MultiDimRandomSearch",
    "MultiDimHillClimber",
    "CMAESAgent",
    "BayesianOptAgent",
    "MultiDimQLearning",
]
