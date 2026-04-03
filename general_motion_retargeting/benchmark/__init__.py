"""
GMR Benchmark: IK Config Auto-Tuning System.

This sub-package provides tools for automatically optimising the IK configuration
of the General Motion Retargeting (GMR) system using black-box optimisation
(Optuna with TPE / CMA-ES / NSGA-II samplers).

Sub-modules
-----------
param_space  : Defines the parameter search space and maps Optuna trial suggestions
               to a modified IK config dict.
dataset_loader : Loads and samples motion sequences from LaFan1 (BVH) and AMASS
               (SMPL-X) datasets for use during optimisation.
evaluator    : Runs retargeting on a batch of sequences and computes quality
               metrics (IK error, smoothness, joint-limit violation rate, root
               trajectory DTW distance).
"""

from .dataset_loader import DatasetLoader
from .evaluator import RetargetingEvaluator
from .param_space import IKConfigParamSpace

__all__ = ["DatasetLoader", "IKConfigParamSpace", "RetargetingEvaluator"]
