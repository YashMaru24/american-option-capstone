"""American put early-exercise environment.

Dependency note: requires `gymnasium` (new dependency, merge into
requirements.txt manually).

Underlying stock dynamics: simulated on the same CRR risk-neutral lattice
used by src/pricing/binomial.py (u, d, p derived from contract.sigma,
contract.r, dt = T/steps) rather than continuous GBM, so that the RL policy
is directly comparable to the binomial exercise boundary benchmark.

State representation (fixed ordering, load-bearing for downstream code):
    state = [step / steps, time_to_maturity_fraction, S / K]
    i.e. [normalized_time_elapsed, normalized_time_remaining, moneyness]

Reward: 0 on every hold action (no reward leakage). On exercise, or on
forced terminal expiry, the reward is the discounted intrinsic payoff,
discounted exactly once by exp(-r * time_elapsed_so_far).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.contract import OptionContract
from src.pricing.payoffs import put_payoff

SEED = 42


class AmericanPutExerciseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        contract: Optional[OptionContract] = None,
        contract_factory: Optional[Callable[[], OptionContract]] = None,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        if contract is None and contract_factory is None:
            raise ValueError("Must supply either contract or contract_factory")

        self._contract = contract
        self._contract_factory = contract_factory
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # 0 = hold, 1 = exercise

        self.contract: OptionContract = None  # type: ignore
        self.dt: float = 0.0
        self.u: float = 0.0
        self.d: float = 0.0
        self.p: float = 0.0
        self.step_idx: int = 0
        self.S: float = 0.0

    def _draw_contract(self) -> OptionContract:
        if self._contract_factory is not None:
            return self._contract_factory()
        return self._contract  # type: ignore

    def _crr_params(self, contract: OptionContract) -> tuple[float, float, float, float]:
        dt = contract.T / contract.steps
        u = math.exp(contract.sigma * math.sqrt(dt))
        d = 1.0 / u
        grow = math.exp(contract.r * dt)
        p = (grow - d) / (u - d)
        return dt, u, d, p

    def _state(self) -> np.ndarray:
        steps = self.contract.steps
        time_elapsed_frac = self.step_idx / steps
        time_remaining_frac = 1.0 - time_elapsed_frac
        moneyness = self.S / self.contract.K
        return np.array([time_elapsed_frac, time_remaining_frac, moneyness], dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.contract = self._draw_contract()
        self.contract.validate()
        self.dt, self.u, self.d, self.p = self._crr_params(self.contract)

        self.step_idx = 0
        self.S = self.contract.S0

        info = {"S": self.S, "step": self.step_idx}
        return self._state(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        contract = self.contract
        time_elapsed = self.step_idx * self.dt

        if action == 1:
            reward = put_payoff(self.S, contract.K) * math.exp(-contract.r * time_elapsed)
            terminated = True
            truncated = False
            info = {"S": self.S, "step": self.step_idx}
            return self._state(), float(reward), terminated, truncated, info

        # action == 0 (hold)
        if self.step_idx >= contract.steps:
            # Should not normally happen (final step forces exercise below),
            # but guard defensively.
            reward = put_payoff(self.S, contract.K) * math.exp(-contract.r * time_elapsed)
            return self._state(), float(reward), True, False, {"S": self.S, "step": self.step_idx}

        # advance one step on the CRR lattice
        up = self._rng.random() < self.p
        self.S = self.S * self.u if up else self.S * self.d
        self.step_idx += 1

        if self.step_idx >= contract.steps:
            # forced terminal exercise / expiry
            time_elapsed_terminal = self.step_idx * self.dt
            reward = put_payoff(self.S, contract.K) * math.exp(-contract.r * time_elapsed_terminal)
            terminated = True
        else:
            reward = 0.0
            terminated = False

        truncated = False
        info = {"S": self.S, "step": self.step_idx}
        return self._state(), float(reward), terminated, truncated, info


def env_factory(
    contract: Optional[OptionContract] = None,
    contract_factory: Optional[Callable[[], OptionContract]] = None,
    seed: int = SEED,
) -> AmericanPutExerciseEnv:
    return AmericanPutExerciseEnv(contract=contract, contract_factory=contract_factory, seed=seed)
