"""MotifCheck: shortcuts in small-model RNA-binding prediction.

A reproducible, leakage-controlled benchmark testing whether balancing
training examples for broad sequence composition and transcript context
improves evidence of motif-specific recognition while preserving prediction
of observed ENCODE PTBP1 eCLIP peaks in K562.

Implements the protocol frozen in ``configs/protocol.yaml``.
"""

__version__ = "1.0.0"

DATASET_VERSION = "motifcheck-ptbp1-k562-grch38-v1"

__all__ = ["__version__", "DATASET_VERSION"]
