from __future__ import annotations

import argparse
import math
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.contract import OptionContract
from src.pricing.binomial import BinomialAmericanPutPricer
from src.rl.env import AmericanPutExerciseEnv, env_factory
from src.rl.train_dqn import QNetwork

SEED = 42

PolicyFn = Callable[[np.ndarray], int]


def always_hold_policy(state: np.ndarray) -> int:
    return 0


def immediate_exercise_policy(state: np.ndarray) -> int:
    return 1


def random_policy(seed: int = SEED) -> PolicyFn:
    rng = np.random.default_rng(seed)

    def policy_fn(state: np.ndarray) -> int:
        return int(rng.integers(0, 2))

    return policy_fn


def binomial_boundary_policy(
    boundary_by_step: dict[int, Optional[float]], K: float, steps: int
) -> PolicyFn:
    def policy_fn(state: np.ndarray) -> int:
        time_elapsed_frac = state[0]
        moneyness = state[2]
        current_step = int(round(time_elapsed_frac * steps))
        S = moneyness * K
        boundary = boundary_by_step.get(current_step)
        if boundary is None:
            return 0
        return 1 if S <= boundary else 0

    return policy_fn


def _load_dqn_policy(path: str) -> PolicyFn:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    q_net = QNetwork(
        state_dim=checkpoint["state_dim"],
        n_actions=checkpoint["n_actions"],
        hidden_layer_sizes=checkpoint["hidden_layer_sizes"],
    )
    q_net.load_state_dict(checkpoint["state_dict"])
    q_net.eval()

    def policy_fn(state: np.ndarray) -> int:
        with torch.no_grad():
            q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    return policy_fn


def evaluate_policy(
    env_factory_fn: Callable[[], AmericanPutExerciseEnv],
    policy_fn: PolicyFn,
    episodes: int = 10_000,
) -> dict:
    rewards = []
    exercised_flags = []
    exercise_steps = []

    for ep in range(episodes):
        env = env_factory_fn()
        state, _ = env.reset(seed=SEED + ep)
        done = False
        exercised = False
        exercise_step = None

        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done and action == 1:
                exercised = True
                exercise_step = info["step"]
            elif done:
                exercise_step = info["step"]

        rewards.append(reward)
        exercised_flags.append(exercised)
        exercise_steps.append(exercise_step)

    rewards_arr = np.array(rewards, dtype=float)
    mean_value = float(rewards_arr.mean())
    std_error = float(rewards_arr.std(ddof=1) / math.sqrt(len(rewards_arr))) if len(rewards_arr) > 1 else 0.0
    exercise_rate = float(np.mean(exercised_flags))
    avg_exercise_step = float(np.mean(exercise_steps)) if exercise_steps else 0.0

    return {
        "value": mean_value,
        "std_error": std_error,
        "exercise_rate": exercise_rate,
        "avg_exercise_step": avg_exercise_step,
    }


def build_policy_comparison(
    env_factory_fn: Callable[[], AmericanPutExerciseEnv],
    policies: dict[str, PolicyFn],
    episodes: int = 10_000,
) -> pd.DataFrame:
    rows = []
    for name, policy_fn in policies.items():
        result = evaluate_policy(env_factory_fn, policy_fn, episodes=episodes)
        rows.append({"policy": name, **result})

    df = pd.DataFrame(rows, columns=["policy", "value", "std_error", "exercise_rate", "avg_exercise_step"])
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def boundary_agreement(
    policy_fn: PolicyFn,
    boundary_by_step: dict[int, Optional[float]],
    steps: int = 100,
    K: float = 100.0,
) -> float:
    money_grid = np.linspace(0.6, 1.4, 81)
    matches = 0
    total = 0

    for step in range(steps + 1):
        boundary = boundary_by_step.get(step)
        for m in money_grid:
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            policy_action = policy_fn(state)

            S = m * K
            binomial_action = 1 if (boundary is not None and S <= boundary) else 0

            if policy_action == binomial_action:
                matches += 1
            total += 1

    return matches / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQN policy against baselines.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    contract = OptionContract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=100, option_type="put")

    binomial_pricer = BinomialAmericanPutPricer()
    result = binomial_pricer.price(contract)
    boundary_by_step = result.metadata["exercise_boundary"]

    def make_env() -> AmericanPutExerciseEnv:
        return env_factory(contract=contract, seed=SEED)

    dqn_policy = _load_dqn_policy(args.model)

    policies: dict[str, PolicyFn] = {
        "dqn": dqn_policy,
        "always_hold": always_hold_policy,
        "immediate_exercise": immediate_exercise_policy,
        "random": random_policy(seed=SEED),
        "binomial_boundary": binomial_boundary_policy(boundary_by_step, K=contract.K, steps=contract.steps),
    }

    comparison_df = build_policy_comparison(make_env, policies, episodes=10_000)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    comparison_df.to_csv(args.out, index=False)

    agreement = boundary_agreement(dqn_policy, boundary_by_step, steps=contract.steps, K=contract.K)

    print(comparison_df.to_string(index=False))
    print(f"DQN boundary agreement vs binomial: {agreement:.4f}")


if __name__ == "__main__":
    main()
