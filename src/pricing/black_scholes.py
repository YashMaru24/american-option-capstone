"""Black-Scholes closed-form pricing.

This module provides the European call/put closed-form price only. It is a
reference price for the European exercise style and does NOT capture the
value of American-style early exercise; American options are always worth at
least as much as their European counterparts.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from src.contract import OptionContract


def black_scholes_price(contract: OptionContract) -> float:
    contract.validate()
    S0, K, T, r, sigma = contract.S0, contract.K, contract.T, contract.r, contract.sigma

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if contract.option_type == "call":
        return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif contract.option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {contract.option_type!r}")
