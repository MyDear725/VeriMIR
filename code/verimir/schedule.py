from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sv_bt5sd_effective_weight(base_weight: float, epoch: int) -> float:
    """Return the frozen two-epoch V140 pulse schedule."""
    if int(epoch) < 1:
        raise ValueError("SV-BT5SD epoch must be >= 1")
    if float(base_weight) < 0.0:
        raise ValueError("SV-BT5SD base weight must be non-negative")
    if int(epoch) == 1:
        return float(base_weight)
    if int(epoch) == 2:
        return 0.5 * float(base_weight)
    return 0.0
