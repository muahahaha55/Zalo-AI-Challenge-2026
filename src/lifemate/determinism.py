"""Fix every source of randomness we control (the contest requires reproducible runs; decoding itself is greedy)."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch when they are installed. Safe to call before any model is loaded.

    This does not force deterministic GPU kernels: whether two runs are byte-identical on the target GPU has to be
    verified by running twice and comparing (see docs/RULES.md section 5.2)."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))      # only effective for child processes
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
