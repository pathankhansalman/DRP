"""DRP Kernel package: self-learning decision graph layer.

The Kernel turns DRP records into trainable graph nodes ("neurons") with
weighted causal inputs, intent metadata, and outcome feedback. The neural
core lives in :mod:`kernel.neural_core` and depends only on the Python
standard library.
"""

from kernel.neural_core import KernelGraph, KernelNeuron, train_on_dataset

__all__ = ["KernelGraph", "KernelNeuron", "train_on_dataset"]
