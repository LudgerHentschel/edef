"""
EDEF: Euler Decomposition of Explained Fit.

Public API
----------
LinearExplainer
TorchExplainer
TreeExplainer
NumericalExplainer
EDEFExplanation
"""

from ._linear import (
    LinearExplainer,
    linear_logistic_components,
    linear_multiclass_components,
    linear_regression_components,
)
from ._results import EDEFExplanation
from ._torch import TorchExplainer
from ._jax import EquinoxExplainer, FlaxExplainer, JaxExplainer, NNXExplainer
from ._tree import TreeExplainer
from ._numerical import NumericalExplainer

__all__ = [
    "EDEFExplanation",
    "LinearExplainer",
    "TorchExplainer",
    "EquinoxExplainer",
    "FlaxExplainer",
    "JaxExplainer",
    "NNXExplainer",
    "TreeExplainer",
    "NumericalExplainer",
    "linear_logistic_components",
    "linear_multiclass_components",
    "linear_regression_components",
]