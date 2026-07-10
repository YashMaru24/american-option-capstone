from __future__ import annotations

import math

import numpy as np
import pytest

from src.contract import OptionContract
from src.pricing.payoffs import put_payoff
from src.rl.env import AmericanPutExerciseEnv, env_factory

SEED = 42


def _contract(**overrides) -> OptionContract:
    defaults = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=10, option_type="put")
    defaults.update(overrides)
    return OptionContract(**defaults)


def test_reset_returns_valid_state():
    env = env_factory(contract=_contract(), seed=SEED)
    state, info = env.reset(seed=SEED)

    assert state.shape == (3,)
    assert state.dtype == np.float32
    assert 0.0 <= state[0] <= 1.0
    assert 0.0 <= state[1] <= 1.0
    assert state[2] > 0.0
    assert "S" in info and "step" in info


def test_exercise_action_terminates_with_discounted_intrinsic_reward():
    contract = _contract()
    env = env_factory(contract=contract, seed=SEED)
    state, _ = env.reset(seed=SEED)

    # Take a few hold steps first, then exercise.
    for _ in range(3):
        state, reward, terminated, truncated, info = env.step(0)
        assert terminated is False
        assert reward == 0.0

    S_current = info["S"]
    step_current = info["step"]
    state, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False

    time_elapsed = step_current * env.dt
    expected_reward = put_payoff(S_current, contract.K) * math.exp(-contract.r * time_elapsed)
    assert reward == pytest.approx(expected_reward, abs=1e-8)


def test_hold_until_final_step_forces_termination_with_correct_discount():
    contract = _contract(steps=5)
    env = env_factory(contract=contract, seed=SEED)
    state, _ = env.reset(seed=SEED)

    terminated = False
    reward = 0.0
    info = {}
    step_count = 0

    while not terminated:
        state, reward, terminated, truncated, info = env.step(0)
        step_count += 1
        assert step_count <= contract.steps

    S_terminal = info["S"]
    time_elapsed_terminal = contract.steps * env.dt
    expected_reward = put_payoff(S_terminal, contract.K) * math.exp(-contract.r * time_elapsed_terminal)

    assert reward == pytest.approx(expected_reward, abs=1e-8)
    assert time_elapsed_terminal == pytest.approx(contract.T, abs=1e-8)


def test_hold_actions_before_terminal_return_zero_reward():
    contract = _contract(steps=8)
    env = env_factory(contract=contract, seed=SEED)
    state, _ = env.reset(seed=SEED)

    for _ in range(contract.steps - 1):
        state, reward, terminated, truncated, info = env.step(0)
        if not terminated:
            assert reward == 0.0


def test_episode_step_count_never_exceeds_contract_steps():
    contract = _contract(steps=6)
    env = env_factory(contract=contract, seed=SEED)
    state, _ = env.reset(seed=SEED)

    terminated = False
    step_count = 0

    while not terminated:
        state, reward, terminated, truncated, info = env.step(0)
        step_count += 1
        assert info["step"] <= contract.steps
        assert step_count <= contract.steps
