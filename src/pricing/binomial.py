from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.contract import OptionContract
from src.pricing.base import Pricer, PricingResult
from src.pricing.payoffs import payoff


def _crr_params(contract: OptionContract) -> tuple[float, float, float, float, float]:
    dt = contract.T / contract.steps
    u = math.exp(contract.sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-contract.r * dt)
    grow = math.exp(contract.r * dt)

    if not (d < grow < u):
        raise ValueError(
            "Arbitrage condition violated: require d < exp(r*dt) < u, "
            f"got d={d}, exp(r*dt)={grow}, u={u}"
        )

    p = (grow - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"Risk-neutral probability out of (0, 1): p={p}")

    return dt, u, d, p, disc


def crr_price(contract: OptionContract) -> float:
    contract.validate()
    n = contract.steps
    dt, u, d, p, disc = _crr_params(contract)

    j = np.arange(n + 1)
    S_terminal = contract.S0 * (u ** j) * (d ** (n - j))
    values = payoff(S_terminal, contract.K, contract.option_type)

    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        S_i = contract.S0 * (u ** j) * (d ** (i - j))
        continuation = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        exercise = payoff(S_i, contract.K, contract.option_type)
        values = np.maximum(continuation, exercise)

    return float(values[0])


def crr_american_put_with_boundary(
    contract: OptionContract,
) -> tuple[float, dict[int, Optional[float]]]:
    contract.validate()
    n = contract.steps
    dt, u, d, p, disc = _crr_params(contract)

    j = np.arange(n + 1)
    S_terminal = contract.S0 * (u ** j) * (d ** (n - j))
    values = payoff(S_terminal, contract.K, "put")

    boundary: dict[int, Optional[float]] = {n: None}

    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        S_i = contract.S0 * (u ** j) * (d ** (i - j))
        continuation = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        exercise = payoff(S_i, contract.K, "put")
        exercised = exercise > continuation
        values = np.maximum(continuation, exercise)

        if np.any(exercised):
            boundary[i] = float(np.max(S_i[exercised]))
        else:
            boundary[i] = None

    return float(values[0]), boundary


class BinomialAmericanPutPricer(Pricer):
    name = "crr_binomial_american_put"

    def price(self, contract: OptionContract) -> PricingResult:
        price, boundary = crr_american_put_with_boundary(contract)
        return PricingResult(price, {"exercise_boundary": boundary})
