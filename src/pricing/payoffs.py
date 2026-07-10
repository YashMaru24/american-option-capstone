from __future__ import annotations

from typing import Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]


def put_payoff(S: ArrayOrFloat, K: float) -> ArrayOrFloat:
    return np.maximum(K - np.asarray(S, dtype=float), 0.0) if isinstance(S, np.ndarray) else max(K - S, 0.0)


def call_payoff(S: ArrayOrFloat, K: float) -> ArrayOrFloat:
    return np.maximum(np.asarray(S, dtype=float) - K, 0.0) if isinstance(S, np.ndarray) else max(S - K, 0.0)


def payoff(S: ArrayOrFloat, K: float, option_type: str) -> ArrayOrFloat:
    if option_type == "put":
        return put_payoff(S, K)
    elif option_type == "call":
        return call_payoff(S, K)
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")
