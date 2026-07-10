# Additional dependencies needed (merge into requirements.txt manually):
#   gymnasium
#   torch
#   pyyaml

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Deque, NamedTuple

import numpy as np
import torch
import yaml
from torch import nn, optim

from src.contract import OptionContract
from src.data.synthetic_contracts import make_contract_grid
from src.rl.env import AmericanPutExerciseEnv, env_factory

SEED = 42


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_layer_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def _make_contract_factory(seed: int) -> callable:
    grid_df = make_contract_grid()
    rng = np.random.default_rng(seed)

    def factory() -> OptionContract:
        row = grid_df.iloc[rng.integers(0, len(grid_df))]
        return OptionContract(
            S0=float(row["S0"]),
            K=float(row["K"]),
            T=float(row["T"]),
            r=float(row["r"]),
            sigma=float(row["sigma"]),
            steps=int(row["steps"]),
            option_type="put",
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN early-exercise policy.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", SEED))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    contract_factory = _make_contract_factory(seed)
    env = env_factory(contract_factory=contract_factory, seed=seed)

    state_dim = 3
    n_actions = 2
    hidden_layer_sizes = list(config["hidden_layer_sizes"])

    q_net = QNetwork(state_dim, n_actions, hidden_layer_sizes)
    target_net = QNetwork(state_dim, n_actions, hidden_layer_sizes)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.MSELoss()

    buffer = ReplayBuffer(int(config["replay_buffer_size"]), seed=seed)

    gamma = float(config["gamma"])
    epsilon = float(config["epsilon_start"])
    epsilon_end = float(config["epsilon_end"])
    epsilon_decay = float(config["epsilon_decay"])
    batch_size = int(config["batch_size"])
    target_update_freq = int(config["target_update_freq"])
    episodes = int(config["episodes"])

    rng = np.random.default_rng(seed)
    running_reward = 0.0
    global_step = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            buffer.push(Transition(state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            global_step += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

                q_pred = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(next_states).max(dim=1).values
                    q_target = rewards + gamma * q_next * (1.0 - dones)

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward if episode > 0 else episode_reward

        if episode % 100 == 0:
            print(f"episode {episode}/{episodes} running_reward={running_reward:.4f} epsilon={epsilon:.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": q_net.state_dict(),
            "hidden_layer_sizes": hidden_layer_sizes,
            "state_dim": state_dim,
            "n_actions": n_actions,
            "config": config,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
